"""Per-task AUPRC/AUROC with 95% CIs from a bootstrap over connected groups (plan 4.1), and the paired
comparison of two models (difference CI, win rate). Resampling whole groups keeps related genes together.

With --matched, negatives are reweighted so that their joint distribution over baseline-expression bins
(quintiles of atlas mean log2 TPM x quintiles of tissue specificity tau, computed on the task's genes;
unexpressed genes form their own bin) matches the positives': the "expression-context-matched" metric.
Positives in bins without any negative are dropped (their share is reported).

Usage: python3 -I evaluate.py <project_dir> <predA.tsv.gz> [<predB.tsv.gz>] [--n 2000] [--matched] [--out out.tsv]
  pred files: columns gene, task, fold, y, pred (out-of-fold test predictions)
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("project"); ap.add_argument("a"); ap.add_argument("b", nargs="?")
ap.add_argument("--n", type=int, default=2000); ap.add_argument("--out")
ap.add_argument("--matched", action="store_true")
args = ap.parse_args()
grp = pd.read_csv(f"{args.project}/splits/groups.tsv", sep="\t").set_index("gene").group
cov = pd.read_csv(f"{args.project}/features/gene_covariates.tsv", sep="\t").set_index("gene")


def match_weights(genes, y):
    """weights making the negatives' (expression level x tau) bin distribution equal the positives'."""
    e = cov.loc[genes, "expr_mean_log2tpm"].values
    t = cov.loc[genes, "expr_tau"].values
    eb = pd.qcut(pd.Series(e).rank(method="first"), 5, labels=False).values
    tb = np.where(np.isnan(t), -1, pd.qcut(pd.Series(np.nan_to_num(t, nan=-1)).rank(method="first"), 5, labels=False).values)
    tb = np.where(np.isnan(t), 5, tb)
    b = eb * 10 + tb
    w = np.zeros(len(y))
    for k in np.unique(b):
        m = b == k
        npos, nneg = y[m].sum(), (1 - y[m]).sum()
        if nneg == 0:
            continue
        w[m & (y == 1)] = 1.0
        w[m & (y == 0)] = (npos / max(y.sum(), 1)) / (nneg / max((1 - y).sum(), 1)) if npos > 0 else 0.0
    return w
A = pd.read_csv(args.a, sep="\t")
B = pd.read_csv(args.b, sep="\t") if args.b else None
rows = []
for task, qa in A.groupby("task"):
    q = qa[["gene", "y", "pred"]].rename(columns={"pred": "pa"})
    if B is not None:
        qb = B[B.task == task][["gene", "pred"]].rename(columns={"pred": "pb"})
        q = q.merge(qb, on="gene", how="inner")
        if len(q) == 0:          # task not covered by model B (e.g. Koh et al. 2026 has no alkaline model)
            continue
    g = q.gene.map(grp).values
    ug, inv = np.unique(g, return_inverse=True)
    y = q.y.values
    w0 = match_weights(q.gene.values, y) if args.matched else np.ones(len(y))
    kept = (w0[y == 1] > 0).mean()
    # positives in bins without negatives get weight 0; renormalise negatives so prevalence stays the positives' share
    rng = np.random.default_rng(0)
    base_a = average_precision_score(y, q.pa, sample_weight=w0)
    base_b = average_precision_score(y, q.pb, sample_weight=w0) if B is not None else np.nan
    bs_a, bs_d = [], []
    for _ in range(args.n):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[inv] * w0
        if (w * y).sum() == 0 or (w * (1 - y)).sum() == 0:
            continue
        a = average_precision_score(y, q.pa, sample_weight=w)
        bs_a.append(a)
        if B is not None:
            bs_d.append(a - average_precision_score(y, q.pb, sample_weight=w))
    lo, hi = np.percentile(bs_a, [2.5, 97.5])
    prev = float((w0 * y).sum() / w0.sum())
    row = dict(task=task, n=len(q), pos=int(y.sum()), matched=args.matched, pos_kept=round(kept, 3),
               prevalence=round(prev, 5), auprc_a=round(base_a, 5), auprc_a_lo=round(lo, 5), auprc_a_hi=round(hi, 5),
               lift_a=round(base_a / prev, 2), auroc_a=round(roc_auc_score(y, q.pa, sample_weight=w0), 4))
    if B is not None:
        dlo, dhi = np.percentile(bs_d, [2.5, 97.5])
        row.update(auprc_b=round(base_b, 5), lift_b=round(base_b / prev, 2), auroc_b=round(roc_auc_score(y, q.pb, sample_weight=w0), 4),
                   diff=round(base_a - base_b, 5), diff_lo=round(dlo, 5), diff_hi=round(dhi, 5),
                   win_rate=round(float(np.mean(np.array(bs_d) > 0)), 3),
                   verdict="A better" if dlo > 0 else ("B better" if dhi < 0 else "not distinguishable"))
    rows.append(row)
res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(res.to_string(index=False))
if args.out:
    res.to_csv(args.out, sep="\t", index=False)
