"""Stack two models' out-of-fold predictions per task (logistic regression on their logits, trained on the
other four folds, applied to the held-out fold): does model B add to model A?

Usage: python3 -I stack_oof.py <predA.tsv.gz> <predB.tsv.gz> <out_predictions.tsv.gz>
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

A, B, OUT = sys.argv[1:4]
a = pd.read_csv(A, sep="\t"); b = pd.read_csv(B, sep="\t")
m = a.merge(b[["gene", "task", "pred"]], on=["gene", "task"], suffixes=("_a", "_b"))
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
out = []
for task, q in m.groupby("task"):
    X = np.c_[lg(q.pred_a.values), lg(q.pred_b.values)]
    y = q.y.values; f = q.fold.values; p = np.zeros(len(q))
    for k in np.unique(f):
        tr, te = f != k, f == k
        p[te] = LogisticRegression(C=1.0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    out.append(pd.DataFrame({"gene": q.gene, "task": task, "fold": f, "y": y, "pred": p}))
pd.concat(out).to_csv(OUT, sep="\t", index=False)
print("wrote", OUT)
