"""In-silico insertion of stress elements into background windows (plan step 6: position, orientation, combinations,
shared vs stress-specific elements), scored with StressCRE (mean of the sequence CNN and the PlantCAD2 head).

Backgrounds: 100 genes drawn at random (seed 0) among genes that are strictly non-responsive for at least three of
drought, salt, heat and cold; each keeps its own window and gene features and is scored by the models of the three
split seeds in which it was in the test fold. Elements (consensus sequences, --elements file: name<TAB>sequence) are
written over the window bases at a given position relative to the TSS (window index 1000 + offset) or the TTS
(index 2019 + offset), on the forward strand or reverse-complemented; constructs whose site would cover a masked (N)
base are skipped for that background. Pairs: two elements 10 bp apart, 5' element at -150.
Effect = change in StressCRE's predicted probability (construct - unmodified window), per task.

Usage: CUDA_VISIBLE_DEVICES=<g> python -I insertion.py <project> <elements.tsv> <model_dir> [--n-bg 100]
Writes results/insertion_effects.tsv.gz (background, construct, element, partner, anchor, offset, strand, task, effect).
Each run overwrites its --out file; insertion_logodds.py merges the runs (--add) and adds p0 (prediction for the
unmodified window) and lor (change in log odds). The paper's runs: jobs/insertion_runs.sh.
"""
import argparse, glob, os, sys, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_models import TASKS, encode, onehot, revcomp_onehot   # noqa: E402
from cre_data import read_windows, labels_matrix, covariates, zscore, load_fold_model   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("project"); ap.add_argument("elements"); ap.add_argument("model_dir")
ap.add_argument("--n-bg", type=int, default=100); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--offsets", default="-900,-700,-500,-350,-250,-150,-100,-50", help="TSS offsets of the single sites")
ap.add_argument("--singles-only", action="store_true", help="only forward single sites at --offsets (no reverse, TTS or pairs)")
ap.add_argument("--out", default="results/insertion_effects.tsv.gz", help="output path relative to the project")
a = ap.parse_args()
P = a.project
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.15)
t0 = time.time()
COMP = str.maketrans("ACGT", "TGCA")
rc = lambda s: s.translate(COMP)[::-1]
names, seqs = read_windows(f"{P}/seqs/wm82a4_windows.fa.gz")
gi = {g: i for i, g in enumerate(names)}
Y = labels_matrix(pd.read_csv(f"{P}/labels/labels_v1.tsv.gz", sep="\t"), names)
C = covariates(f"{P}/features/gene_covariates.tsv", names)
folds = {s: pd.read_csv(f"{P}/splits/fold_seed{s}.tsv", sep="\t").set_index("gene").fold.reindex(names).values for s in range(3)}
up_cols = [TASKS.index(f"{s}-up") for s in ("drought", "salt", "heat", "cold")]
dn_cols = [TASKS.index(f"{s}-down") for s in ("drought", "salt", "heat", "cold")]
nonresp = ((Y[:, up_cols] == 0) & (Y[:, dn_cols] == 0)).sum(1) >= 3
bg = np.sort(np.random.default_rng(0).choice(np.where(nonresp)[0], a.n_bg, replace=False))
el = pd.read_csv(a.elements, sep="\t", header=None, names=["name", "seq"]).set_index("name").seq.to_dict()
OFFS = [int(x) for x in a.offsets.split(",")]
constructs = []      # (construct id, element, partner, anchor, offset, strand, [(index, sequence), ...])
for e, s in el.items():
    for o in OFFS:
        constructs.append((f"{e}|tss{o}|+", e, "", "tss", o, "+", [(1000 + o, s)]))
    if a.singles_only:
        continue
    constructs.append((f"{e}|tss-150|-", e, "", "tss", -150, "-", [(850, rc(s))]))
    for o in (200, 500):
        constructs.append((f"{e}|tts+{o}|+", e, "", "tts", o, "+", [(2019 + o, s)]))
names_el = [] if a.singles_only else list(el)
for i, e1 in enumerate(names_el):
    for e2 in names_el[i:]:
        s1, s2 = el[e1], el[e2]
        constructs.append((f"{e1}+{e2}|tss-150|+", e1, e2, "tss", -150, "+", [(850, s1), (850 + len(s1) + 10, s2)]))
print(f"{len(bg)} backgrounds, {len(constructs)} constructs", flush=True)

from transformers import AutoModel, AutoTokenizer   # noqa: E402
tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
lm = AutoModel.from_pretrained(a.model_dir, trust_remote_code=True).to(dev).eval()


@torch.no_grad()
def embed(ss):
    out = []
    for s0 in range(0, len(ss), a.batch):
        ids = tok(ss[s0:s0 + a.batch], return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = lm(input_ids=ids).last_hidden_state.float()
        out.append(torch.cat([h[:, :1500].reshape(h.shape[0], 30, 50, -1).mean(2), h[:, 1520:].reshape(h.shape[0], 30, 50, -1).mean(2)], 1))
    return torch.cat(out)


models = {(kind, s, k): load_fold_model(f"{P}/models/v1_{kind}_main_seed{s}/fold{k}.pt", C.shape[1], dev)
          for kind in ("cnn", "pc2") for s in range(3) for k in range(5)}
rows = []
for b in bg:
    ref = seqs[b]
    ss, meta = [ref], [None]
    for cid, e, e2, anc, o, st, ins in constructs:
        x = ref
        ok = True
        for idx, s in ins:
            if "N" in ref[idx:idx + len(s)] or idx + len(s) > len(ref):
                ok = False
                break
            x = x[:idx] + s + x[idx + len(s):]
        if ok:
            ss.append(x); meta.append((cid, e, e2, anc, o, st))
    with torch.no_grad():
        xo = onehot(torch.from_numpy(np.stack([encode(s) for s in ss])).to(dev))
        xe = embed(ss)
        p = np.zeros((len(ss), len(TASKS)), np.float32)
        for s in range(3):
            k = int(folds[s][b])
            for kind in ("cnn", "pc2"):
                m, mu, sd, _ = models[(kind, s, k)]
                c = torch.from_numpy(zscore(C[b:b + 1], mu, sd)).to(dev).expand(len(ss), -1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = 0.5 * (m(xo, c).float() + m(revcomp_onehot(xo), c).float()) if kind == "cnn" else m(xe, c).float()
                p += torch.sigmoid(z).cpu().numpy() / 6
    d = p[1:] - p[0]
    for (cid, e, e2, anc, o, st), dd in zip(meta[1:], d):
        for j, t in enumerate(TASKS):
            rows.append((names[b], cid, e, e2, anc, o, st, t, round(float(dd[j]), 6)))
    if len(rows) % 20000 < len(TASKS) * len(constructs):
        print(f"{names[b]} done [{(time.time() - t0) / 60:.1f} min]", flush=True)
pd.DataFrame(rows, columns=["background", "construct", "element", "partner", "anchor", "offset", "strand", "task", "effect"]).to_csv(
    f"{P}/{a.out}", sep="\t", index=False)
print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
