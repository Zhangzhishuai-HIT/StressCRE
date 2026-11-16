"""Predicted effect of panel SNPs on the stress response (plan step 8): delta = p(alternative) - p(reference)
for the ten tasks, for one StressCRE component at a time.

Windows: the a2 windows of the genes (variants/panel_snps.tsv.gz from variant_sites.py; idx = window position,
minus-strand windows reverse-complemented). Each a4 gene is scored by the models of the three split seeds in which
it was in the test fold, with its own gene features fixed; deltas are averaged over the three models.
  --component cnn : the sequence CNN (forward/reverse-complement average), every SNP;
  --component pc2 : PlantCAD2-Small representations of the reference and alternative windows are recomputed
                    (last layer, 50-bp bin means, as embed_plantcad2.py) and passed to the PlantCAD2 head; only the
                    SNPs listed in --subset (row numbers of panel_snps.tsv.gz, one per line).
Output: variants/delta_<component>.npy, float16 [n_rows, 10] (NaN for rows not scored), row order of panel_snps.

Usage: CUDA_VISIBLE_DEVICES=<g> python -I score_variants.py <project> --component cnn|pc2 [--subset rows.txt]
         [--model-dir PlantCAD2 dir] [--shard i/n] [--batch 64]
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_models import TASKS, encode, onehot, revcomp_onehot   # noqa: E402
from cre_data import read_windows, covariates, zscore, load_fold_model   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("project"); ap.add_argument("--component", choices=["cnn", "pc2"], required=True)
ap.add_argument("--subset"); ap.add_argument("--model-dir"); ap.add_argument("--shard", default="0/1")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--mem-frac", type=float, default=0.15)
a = ap.parse_args()
P = a.project
SI, SN = (int(x) for x in a.shard.split("/"))
dev = torch.device("cuda")
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(a.mem_frac)          # the cards are shared with other projects
t0 = time.time()
snp = pd.read_csv(f"{P}/variants/panel_snps.tsv.gz", sep="\t")
rows = np.arange(len(snp)) if not a.subset else np.loadtxt(a.subset, dtype=np.int64)
names4, _ = read_windows(f"{P}/seqs/wm82a4_windows.fa.gz")
gi4 = {g: i for i, g in enumerate(names4)}
C = covariates(f"{P}/features/gene_covariates.tsv", names4)
folds = {s: pd.read_csv(f"{P}/splits/fold_seed{s}.tsv", sep="\t").set_index("gene").fold.reindex(names4).values for s in range(3)}
names2, seqs2 = read_windows(f"{P}/seqs/wm82a2_windows.fa.gz")
win2 = dict(zip(names2, seqs2))
comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
sub = snp.iloc[rows].copy()
sub["row"] = rows
sub["a4i"] = sub.gene_a4.map(gi4)
sub = sub[sub.a4i.notna()]
genes = sorted(sub.gene_a4.unique())
genes = [g for k, g in enumerate(genes) if k % SN == SI]
sub = sub[sub.gene_a4.isin(set(genes))]
print(f"{len(sub)} SNP rows, {len(genes)} genes, component {a.component}, shard {SI}/{SN}", flush=True)

if a.component == "pc2":
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    lm = AutoModel.from_pretrained(a.model_dir, trust_remote_code=True).to(dev).eval()

    @torch.no_grad()
    def embed(seqs):
        out = []
        for s0 in range(0, len(seqs), a.batch):
            ids = tok(seqs[s0:s0 + a.batch], return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = lm(input_ids=ids).last_hidden_state.float()
            tss = h[:, :1500].reshape(h.shape[0], 30, 50, h.shape[2]).mean(2)
            tts = h[:, 1520:].reshape(h.shape[0], 30, 50, h.shape[2]).mean(2)
            out.append(torch.cat([tss, tts], 1))
        return torch.cat(out)

delta = np.full((len(snp), len(TASKS)), np.nan, np.float32)
acc = np.zeros((len(snp), len(TASKS)), np.float32)
by_gene = {g: q for g, q in sub.groupby("gene_a4")}
models = {}


def get_model(s, k):
    key = (s, k)
    if key not in models:                      # all 15 fold models stay loaded (small heads)
        models[key] = load_fold_model(f"{P}/models/v1_{a.component}_main_seed{s}/fold{k}.pt", C.shape[1], dev)
    return models[key]


done = 0
for g in genes:
    q = by_gene[g]
    gi = gi4[g]
    ref = win2[q.gene_a2.iloc[0]]
    alts = []
    for r in q.itertuples():
        b = r.alt if r.strand == "+" else comp[r.alt]
        alts.append(ref[:r.idx] + b + ref[r.idx + 1:])
    seqs = [ref] + alts
    with torch.no_grad():
        if a.component == "cnn":
            x = onehot(torch.from_numpy(np.stack([encode(s) for s in seqs])).to(dev))
        else:
            x = embed(seqs)
    for s in range(3):
        model, mu, sd, _ = get_model(s, int(folds[s][gi]))
        c = torch.from_numpy(zscore(C[gi:gi + 1], mu, sd)).to(dev)
        ps = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s0 in range(0, len(seqs), 512):
                xb = x[s0:s0 + 512]
                cb = c.expand(len(xb), -1)
                if a.component == "cnn":
                    z = 0.5 * (model(xb, cb).float() + model(revcomp_onehot(xb), cb).float())
                else:
                    z = model(xb, cb).float()
                ps.append(torch.sigmoid(z).cpu().numpy())
        p = np.concatenate(ps)
        acc[q.row.values] += (p[1:] - p[0]) / 3
    done += len(q)
    if done % 200000 < len(q):
        print(f"{done} SNP rows  [{(time.time() - t0) / 60:.1f} min]", flush=True)
delta[sub.row.values] = acc[sub.row.values]
suffix = "" if SN == 1 else f".shard{SI}"
np.save(f"{P}/variants/delta_{a.component}{suffix}.npy", delta.astype(np.float16))
print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
