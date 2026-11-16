"""StressCRE predictions for every gene and for the WGD paralog pairs with shared gene features (plan step 7).

StressCRE = mean of the sequence-CNN and PlantCAD2-head probabilities. Every gene is predicted by the models of
the three split seeds in which it was in the test fold (a WGD pair is always in one fold), and the three
predictions are averaged.
  1. own gene features: results/v1_stresscre_allgenes.tsv.gz (gene, task, p, p_cnn, p_pc2) for all 52,872 genes;
  2. WGD pairs with shared gene features: both copies get the mean of the two copies' (untransformed) features,
     so the difference between their predictions comes from the sequence windows only:
     results/v1_pairs_sharedfeat.tsv.gz (gene_a, gene_b, task, pa, pb).

Usage: CUDA_VISIBLE_DEVICES=<g> python -I predict_all.py <project> <pairs.tsv>
"""
import glob, os, sys, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_models import TASKS, encode, onehot, revcomp_onehot   # noqa: E402
from cre_data import read_windows, covariates, zscore, load_fold_model   # noqa: E402

P, PAIRS = sys.argv[1:3]
dev = torch.device("cuda")
torch.set_num_threads(4)
t0 = time.time()
names, seqs = read_windows(f"{P}/seqs/wm82a4_windows.fa.gz")
N = len(names)
gi = {g: i for i, g in enumerate(names)}
codes = torch.from_numpy(np.stack([encode(s) for s in seqs])).to(dev)
del seqs
chunks = sorted(glob.glob(f"{P}/embeddings/plantcad2_small_wm82a4/chunk_????.npy"))
E, r0 = None, 0
for c in chunks:
    z = np.load(c, mmap_mode="r")
    if E is None:
        E = np.empty((N,) + z.shape[1:], np.float16)
    E[r0:r0 + len(z)] = z
    r0 += len(z)
assert r0 == N
C = covariates(f"{P}/features/gene_covariates.tsv", names)
pairs = pd.read_csv(PAIRS, sep="\t")
pa = pairs.gene_a.map(gi).values
pb = pairs.gene_b.map(gi).values
Cs = (C[pa] + C[pb]) / 2                                   # shared features per pair
folds = {s: pd.read_csv(f"{P}/splits/fold_seed{s}.tsv", sep="\t").set_index("gene").fold.reindex(names).values for s in range(3)}
for s in range(3):
    assert (folds[s][pa] == folds[s][pb]).all(), "pair split across folds"
print(f"{N} genes, {len(pairs)} pairs; inputs loaded in {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def run(model, kind, idx, cz):
    out = []
    for s0 in range(0, len(idx), 256):
        b = idx[s0:s0 + 256]
        c = torch.from_numpy(cz[s0:s0 + 256]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if kind == "cnn":
                x = onehot(codes[torch.from_numpy(b).to(dev)])
                z = 0.5 * (model(x, c).float() + model(revcomp_onehot(x), c).float())
            else:
                z = model(torch.from_numpy(E[b]).to(dev).float(), c).float()
        out.append(torch.sigmoid(z).cpu().numpy())
    return np.concatenate(out)


own = {k: np.zeros((N, len(TASKS)), np.float32) for k in ("cnn", "pc2")}
sh_a = {k: np.zeros((len(pairs), len(TASKS)), np.float32) for k in ("cnn", "pc2")}
sh_b = {k: np.zeros((len(pairs), len(TASKS)), np.float32) for k in ("cnn", "pc2")}
for kind in ("cnn", "pc2"):
    for s in range(3):
        for k in range(5):
            model, mu, sd, kk = load_fold_model(f"{P}/models/v1_{kind}_main_seed{s}/fold{k}.pt", C.shape[1], dev)
            assert kk == kind
            idx = np.where(folds[s] == k)[0]
            own[kind][idx] += run(model, kind, idx, zscore(C[idx], mu, sd)) / 3
            pi = np.where(folds[s][pa] == k)[0]
            czs = zscore(Cs[pi], mu, sd)
            sh_a[kind][pi] += run(model, kind, pa[pi], czs) / 3
            sh_b[kind][pi] += run(model, kind, pb[pi], czs) / 3
        print(f"{kind} seed {s} done  [{time.time() - t0:.0f}s]", flush=True)
rows = []
for j, t in enumerate(TASKS):
    rows.append(pd.DataFrame({"gene": names, "task": t, "p": ((own["cnn"][:, j] + own["pc2"][:, j]) / 2).round(6),
                              "p_cnn": own["cnn"][:, j].round(6), "p_pc2": own["pc2"][:, j].round(6)}))
pd.concat(rows).to_csv(f"{P}/results/v1_stresscre_allgenes.tsv.gz", sep="\t", index=False)
rows = []
for j, t in enumerate(TASKS):
    rows.append(pd.DataFrame({"gene_a": pairs.gene_a, "gene_b": pairs.gene_b, "task": t,
                              "pa": ((sh_a["cnn"][:, j] + sh_a["pc2"][:, j]) / 2).round(6),
                              "pb": ((sh_b["cnn"][:, j] + sh_b["pc2"][:, j]) / 2).round(6)}))
pd.concat(rows).to_csv(f"{P}/results/v1_pairs_sharedfeat.tsv.gz", sep="\t", index=False)
print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
