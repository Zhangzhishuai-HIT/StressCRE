"""Base-resolution contribution scores of StressCRE's sequence CNN (plan step 6), for motif discovery with
tfmodisco-lite.

For every task, the genes are its strict positives plus up to --neg randomly drawn non-responsive genes (seed 0).
Each gene is scored by the three CNN models in which it was in the test fold (one per split seed of the main
test; the gene's own z-scored gene features are kept fixed), and the scores are averaged.
Method: integrated gradients from dinucleotide-shuffled references (DeepLIFT/DeepSHAP-style hypothetical
contributions): for each of R references (every stretch of A/C/G/T shuffled with its dinucleotide composition
kept; masked coding sequence and spacer stay N), the gradient of the task logit is averaged over S points on the
straight path from the reference to the sequence; hypothetical contribution of base b at position i =
multiplier(i, b) - multiplier(i, reference base at i); averaged over references. The logit is the
forward/reverse-complement average the model uses for prediction. Actual contribution = hypothetical x one-hot.

Usage: CUDA_VISIBLE_DEVICES=<g> python -I attribution.py <project> <out_dir> [--refs 8] [--steps 8] [--neg 1000] [--tasks t1,t2]
Writes <out_dir>/<task>.npz: genes, y, pred, codes [n, L] uint8, hyp [n, 4, L] float16.
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
import torch
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_models import TASKS, encode, onehot, revcomp_onehot   # noqa: E402
from cre_data import read_windows, labels_matrix, covariates, zscore, load_fold_model   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("project"); ap.add_argument("out")
ap.add_argument("--refs", type=int, default=8); ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--neg", type=int, default=1000); ap.add_argument("--tasks", default="")
ap.add_argument("--genes-per-batch", type=int, default=8)
a = ap.parse_args()
P = a.project
os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda")
torch.set_num_threads(4)


@njit(cache=True)
def _dinuc(seq, perm_seed):
    np.random.seed(perm_seed)
    n = len(seq)
    if n < 3:
        return seq.copy()
    nxt = [np.where(seq[:-1] == t)[0] + 1 for t in range(4)]
    for t in range(4):
        m = len(nxt[t])
        if m > 1:
            p = np.random.permutation(m - 1)
            tmp = nxt[t].copy()
            for j in range(m - 1):
                nxt[t][j] = tmp[p[j]]
    cnt = np.zeros(4, np.int64)
    out = np.empty(n, seq.dtype)
    ind = 0
    out[0] = seq[0]
    for j in range(1, n):
        t = seq[ind]
        ind = nxt[t][cnt[t]]
        cnt[t] += 1
        out[j] = seq[ind]
    return out


def shuffle_ref(codes, seed):
    """dinucleotide shuffle of every A/C/G/T stretch; code 4 (N) positions are kept."""
    out = codes.copy()
    ok = codes < 4
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]] + 1
    for k, (s, e) in enumerate(zip(starts, ends)):
        out[s:e] = _dinuc(codes[s:e].astype(np.int64), seed * 1000 + k).astype(codes.dtype)
    return out


t_start = time.time()
names, seqs = read_windows(f"{P}/seqs/wm82a4_windows.fa.gz")
N, L = len(names), len(seqs[0])
codes_all = np.stack([encode(s) for s in seqs])
del seqs
Y = labels_matrix(pd.read_csv(f"{P}/labels/labels_v1.tsv.gz", sep="\t"), names)
C = covariates(f"{P}/features/gene_covariates.tsv", names)
folds = {s: pd.read_csv(f"{P}/splits/fold_seed{s}.tsv", sep="\t").set_index("gene").fold.reindex(names).values for s in range(3)}
tasks = a.tasks.split(",") if a.tasks else TASKS
rng = np.random.default_rng(0)
sel = {}
for t in tasks:
    j = TASKS.index(t)
    pos = np.where(Y[:, j] == 1)[0]
    neg = np.where(Y[:, j] == 0)[0]
    neg = np.sort(rng.choice(neg, min(a.neg, len(neg)), replace=False))
    sel[t] = (np.r_[pos, neg], np.r_[np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)])
print(f"{N} genes; tasks {tasks}; pairs {sum(len(v[0]) for v in sel.values())}; refs {a.refs} x steps {a.steps}", flush=True)

alphas = torch.tensor((np.arange(a.steps) + 0.5) / a.steps, dtype=torch.float32, device=dev)
hyp_acc = {t: np.zeros((len(sel[t][0]), 4, L), np.float32) for t in tasks}
pred_acc = {t: np.zeros(len(sel[t][0]), np.float32) for t in tasks}
for s in range(3):
    for k in range(5):
        model, mu, sd, kind = load_fold_model(f"{P}/models/v1_cnn_main_seed{s}/fold{k}.pt", C.shape[1], dev)
        assert kind == "cnn"
        Cz = zscore(C, mu, sd)
        for t in tasks:
            j = TASKS.index(t)
            gi, _ = sel[t]
            rows = np.where(folds[s][gi] == k)[0]
            for b0 in range(0, len(rows), a.genes_per_batch):
                rb = rows[b0:b0 + a.genes_per_batch]
                g = gi[rb]
                x = onehot(torch.from_numpy(codes_all[g]).to(dev))                          # [B, 4, L]
                refs = np.stack([[shuffle_ref(codes_all[gg], 7919 * r + int(gg)) for r in range(a.refs)] for gg in g])
                rc_ = onehot(torch.from_numpy(refs.reshape(-1, L)).to(dev)).view(len(g), a.refs, 4, L)
                rc_ = rc_ * (x.sum(1, keepdim=True).unsqueeze(1) > 0)                      # keep N columns at zero
                path = rc_.unsqueeze(2) + alphas.view(1, 1, -1, 1, 1) * (x.unsqueeze(1).unsqueeze(2) - rc_.unsqueeze(2))
                inp = path.reshape(-1, 4, L).requires_grad_(True)                          # [B*R*S, 4, L]
                cz = torch.from_numpy(Cz[g]).to(dev).repeat_interleave(a.refs * a.steps, 0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = 0.5 * (model(inp, cz)[:, j].float() + model(revcomp_onehot(inp), cz)[:, j].float())
                grad, = torch.autograd.grad(z.sum(), inp)
                mult = grad.view(len(g), a.refs, a.steps, 4, L).mean(2)                    # [B, R, 4, L]
                at_ref = (mult * rc_).sum(2, keepdim=True)                                 # multiplier at the reference base
                hyp = (mult - at_ref).mean(1)                                              # [B, 4, L]
                hyp_acc[t][rb] += hyp.float().cpu().numpy()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    cg = torch.from_numpy(Cz[g]).to(dev)
                    pz = 0.5 * (model(x, cg)[:, j].float() + model(revcomp_onehot(x), cg)[:, j].float())
                pred_acc[t][rb] += torch.sigmoid(pz).cpu().numpy()
        print(f"seed {s} fold {k} done  [{(time.time() - t_start) / 60:.1f} min]", flush=True)
        del model
for t in tasks:
    gi, y = sel[t]
    np.savez_compressed(os.path.join(a.out, f"{t}.npz"), genes=np.array([names[i] for i in gi]), y=y,
                        pred=(pred_acc[t] / 3).astype(np.float32), codes=codes_all[gi],
                        hyp=(hyp_acc[t] / 3).astype(np.float16))
    print(f"{t}: {len(gi)} genes ({int(y.sum())} positives) written", flush=True)
print(f"done in {(time.time() - t_start) / 60:.1f} min", flush=True)
