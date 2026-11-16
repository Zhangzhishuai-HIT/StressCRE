"""Comparison tables for plan steps 4-5 from out-of-fold prediction files (labels v1, main split, 3 seeds).

1. Choosing the candidate inside the training folds (plan 2.4): expression-matched AUPRC on the inner folds of
   candidate 1 (cnn), candidate 2 (pc2) and candidate 3 (mean of the two predictions), per task and seed
   -> results/v1_candidates_innerval.tsv. The candidate with the highest mean over the 9 main tasks and the
   3 seeds is StressCRE; the test folds play no part in the choice. Candidate 3's prediction files
   (results/v1_cand3_main_seed<s>_{predictions,innerval}.tsv.gz) are written here for evaluate.py.
2. Test folds: matched and plain AUPRC per task for every model and seed, with mean and SD over seeds
   -> results/v1_models_test_main.tsv (long) and results/v1_models_test_main_wide.tsv.
Matched weights as in evaluate.py --matched (cre_eval.py: negatives reweighted to the positives' expression-level x tau mix).

Usage: python -I summarize_models.py <project> [model ...]     (default models: see MODELS)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_eval import scores as _scores   # noqa: E402

P = sys.argv[1]
MODELS = sys.argv[2:] or ["cnn", "pc2", "cand3", "cand3_seqonly", "cov", "kmer", "stack", "deepcre", "motif", "koh", "expr"]
MAIN = [f"{s}-{d}" for s in ["drought", "salt", "heat", "cold", "alkaline"] for d in ("up", "down")]
MAIN.remove("alkaline-down")
cov = pd.read_csv(f"{P}/features/gene_covariates.tsv", sep="\t").set_index("gene")


def scores(df):
    return _scores(cov, df)


def load(model, seed, kind="predictions"):
    if model.startswith("cand3"):              # cand3 = mean of cnn and pc2; cand3_seqonly = the same without gene features
        sfx = model[len("cand3"):]
        a, b = load("cnn" + sfx, seed, kind), load("pc2" + sfx, seed, kind)
        if a is None or b is None:
            return None
        m = a.merge(b[["gene", "task", "pred"]], on=["gene", "task"], suffixes=("", "_b"))
        assert len(m) == len(a) == len(b)
        m = m.assign(pred=(m.pred + m.pred_b) / 2).drop(columns="pred_b")
        f = f"{P}/results/v1_{model}_main_seed{seed}_{kind}.tsv.gz"       # written for evaluate.py
        if not os.path.exists(f):
            m.to_csv(f, sep="\t", index=False)
        return m
    if model == "expr":        # reference row: rank by atlas expression level (higher = more likely responsive)
        ref = load("cov", seed)
        return None if ref is None else ref.assign(pred=cov.loc[ref.gene, "expr_mean_log2tpm"].values)
    f = f"{P}/results/v1_{model}_main_seed{seed}_{kind}.tsv.gz"
    return pd.read_csv(f, sep="\t") if os.path.exists(f) else None


# 1. inner folds
rows = []
for m in ("cnn", "pc2", "cand3"):
    for s in range(3):
        d = load(m, s, "innerval")
        if d is None:
            continue
        for task, v in scores(d).items():
            rows.append(dict(candidate=m, seed=s, task=task, **{k: round(x, 5) for k, x in v.items()}))
iv = pd.DataFrame(rows)
if len(iv):
    iv.to_csv(f"{P}/results/v1_candidates_innerval.tsv", sep="\t", index=False)
    t = iv[iv.task.isin(MAIN)].groupby(["candidate", "seed"]).auprc_matched.mean().unstack()
    t["mean"] = t.mean(1)
    print("inner folds, matched AUPRC, mean over the 9 main tasks:\n" + t.round(4).to_string() + "\n")

# 2. test folds
rows = []
for m in MODELS:
    for s in range(3):
        d = load(m, s)
        if d is None:
            continue
        for task, v in scores(d).items():
            rows.append(dict(model=m, seed=s, task=task, **{k: round(x, 5) for k, x in v.items()}))
te = pd.DataFrame(rows)
te.to_csv(f"{P}/results/v1_models_test_main.tsv", sep="\t", index=False)
g = te.groupby(["model", "task"])
w = g.auprc_matched.mean().unstack(0)
w_sd = g.auprc_matched.std().unstack(0)
prev = te.groupby("task").prevalence_matched.mean()
lift = w.div(prev, axis=0)
out = w.round(4).astype(str) + " ±" + w_sd.round(4).astype(str) + " (" + lift.round(2).astype(str) + "×)"
out = out[[m for m in MODELS if m in out.columns]]
out.to_csv(f"{P}/results/v1_models_test_main_wide.tsv", sep="\t")
pd.set_option("display.width", 250)
print("test folds, matched AUPRC (mean ±SD over seeds; lift = AUPRC / matched prevalence):\n" + out.to_string())
mm = te[te.task.isin(MAIN)].groupby(["model", "seed"]).auprc_matched.mean().unstack()
mm["mean"] = mm.mean(1)
print("\ntest folds, matched AUPRC, mean over the 9 main tasks:\n" + mm.round(4).to_string())
