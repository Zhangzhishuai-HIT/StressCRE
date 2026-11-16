"""Shared evaluation helpers: expression-context-matched AUPRC (same weights as evaluate.py --matched).

match_weights(cov, genes, y): negatives reweighted so that their joint distribution over quintiles of atlas
mean log2 TPM x quintiles of tau (computed on the given genes; genes without tau form their own bin) equals the
positives'; positives in bins without negatives get weight 0.
scores(cov, df): per task of an out-of-fold prediction table (gene, task, fold, y, pred): plain and matched AUPRC
and prevalence.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

MAIN9 = [f"{s}-{d}" for s in ["drought", "salt", "heat", "cold", "alkaline"] for d in ("up", "down")]
MAIN9.remove("alkaline-down")


def match_weights(cov, genes, y):
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


def scores(cov, df):
    out = {}
    for task, q in df.groupby("task"):
        y = q.y.values
        w = match_weights(cov, q.gene.values, y)
        out[task] = dict(auprc=average_precision_score(y, q.pred), prevalence=y.mean(),
                         auprc_matched=average_precision_score(y, q.pred, sample_weight=w),
                         prevalence_matched=(w * y).sum() / w.sum())
    return out
