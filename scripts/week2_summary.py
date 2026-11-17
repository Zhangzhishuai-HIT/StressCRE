"""Summary tables of model accuracy (written during week 2 of the project; used by Figs 1 and 3 and the README).

Reads results/v1_models_test_main.tsv and results/v1_candidates_innerval.tsv (summarize_models.py) and the paired
comparisons results/eval_v1_<chosen>_vs_<ref>_seed<s>.tsv (evaluate.py --matched --n 1000).
Writes results/week2_summary.tsv: per task, matched lift (AUPRC / matched prevalence, mean over the 3 seeds) of
every model, the chosen candidate's matched-AUPRC difference to the stack (mean over seeds) and to the
covariates, and how many of the 3 seeds the paired bootstrap calls "chosen better".
Also results/week2_tests.tsv: mean matched AUPRC over the 9 main tasks of every model on the three tests
(main split: mean over the 3 seeds; chromosome hold-out with homologues removed; random split), and
results/week2_selection.tsv, results/week2_mean9.tsv, and results/week2_koh.tsv (Koh et al. 2026 adapters against
StressCRE and the stack on the 8 tasks Koh covers). Candidate 3's chrom/random prediction files are written here.

Usage: python -I week2_summary.py <project> <chosen_candidate>
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_eval import MAIN9, scores   # noqa: E402

P, CH = sys.argv[1:3]
te = pd.read_csv(f"{P}/results/v1_models_test_main.tsv", sep="\t")
te["lift"] = te.auprc_matched / te.prevalence_matched
lift = te.groupby(["task", "model"]).lift.mean().unstack()
rows = []
for task in lift.index:
    r = dict(task=task)
    for m in lift.columns:
        r[f"lift_{m}"] = round(lift.loc[task, m], 2)
    for ref in ("stack", "cov"):
        d, better, worse = [], 0, 0
        for s in range(3):
            f = f"{P}/results/eval_v1_{CH}_vs_{ref}_seed{s}.tsv"
            if not os.path.exists(f):
                continue
            e = pd.read_csv(f, sep="\t").set_index("task")
            d.append(e.loc[task, "diff"])
            better += e.loc[task, "verdict"] == "A better"
            worse += e.loc[task, "verdict"] == "B better"
            if s == 0:
                r[f"n_{ref}"], r[f"pos_{ref}"] = int(e.loc[task, "n"]), int(e.loc[task, "pos"])
        if d:
            r[f"diff_vs_{ref}"] = round(float(np.mean(d)), 4)
            r[f"seeds_better_vs_{ref}"] = int(better)
            r[f"seeds_worse_vs_{ref}"] = int(worse)
    rows.append(r)
out = pd.DataFrame(rows)
iv = pd.read_csv(f"{P}/results/v1_candidates_innerval.tsv", sep="\t")
main = [t for t in out.task if t != "alkaline-down"]
sel = iv[iv.task.isin(main)].groupby("candidate").auprc_matched.mean().round(4).to_dict()
out.to_csv(f"{P}/results/week2_summary.tsv", sep="\t", index=False)
with open(f"{P}/results/week2_selection.tsv", "w") as f:
    f.write("candidate\tinner_val_matched_auprc_mean9\tchosen\n")
    for c, v in sorted(sel.items(), key=lambda x: -x[1]):
        f.write(f"{c}\t{v}\t{int(c == CH)}\n")
mm = te[te.task.isin(main)].groupby(["model", "seed"]).auprc_matched.mean().groupby("model").agg(["mean", "std"]).round(4)
mm.to_csv(f"{P}/results/week2_mean9.tsv", sep="\t")
cov = pd.read_csv(f"{P}/features/gene_covariates.tsv", sep="\t").set_index("gene")


def load_split(model, split):
    if model == "cand3":
        a, b = load_split("cnn", split), load_split("pc2", split)
        if a is None or b is None:
            return None
        m = a.merge(b[["gene", "task", "pred"]], on=["gene", "task"], suffixes=("", "_b"))
        assert len(m) == len(a) == len(b)
        m = m.assign(pred=(m.pred + m.pred_b) / 2).drop(columns="pred_b")
        f = f"{P}/results/v1_cand3_{split}_predictions.tsv.gz"
        if not os.path.exists(f):
            m.to_csv(f, sep="\t", index=False)
        return m
    f = f"{P}/results/v1_{model}_{split}_predictions.tsv.gz"
    return pd.read_csv(f, sep="\t") if os.path.exists(f) else None


tests = []
for m in ["cov", "kmer", "stack", "deepcre", "motif", "cnn", "pc2", "cand3"]:     # koh: 8 tasks only, see below
    r = dict(model=m)
    q = te[(te.model == m) & te.task.isin(MAIN9)]
    if len(q):
        r["main"] = round(q.groupby("seed").auprc_matched.mean().mean(), 4)
        r["main_sd"] = round(q.groupby("seed").auprc_matched.mean().std(), 4)
    for sp in ("chrom", "random"):
        d = load_split(m, sp)
        if d is not None:
            sc = scores(cov, d)
            r[sp] = round(float(np.mean([sc[t]["auprc_matched"] for t in MAIN9 if t in sc])), 4)
    tests.append(r)
tests = pd.DataFrame(tests)
tests.to_csv(f"{P}/results/week2_tests.tsv", sep="\t", index=False)

# Koh et al. 2026 cross-species reference: only drought/salt/heat/cold x up/down have a model (8 tasks)
K8 = [t for t in MAIN9 if not t.startswith("alkaline")]
kr = []
for m in ["koh", "stack", "cand3"]:
    r = dict(model=m)
    q = te[(te.model == m) & te.task.isin(K8)]
    if len(q):
        r["main"] = round(q.groupby("seed").auprc_matched.mean().mean(), 4)
        r["main_auroc"] = np.nan
    for sp in ("chrom", "random"):
        d = load_split(m, sp)
        if d is not None:
            sc = scores(cov, d)
            r[sp] = round(float(np.mean([sc[t]["auprc_matched"] for t in K8 if t in sc])), 4)
    kr.append(r)
kr = pd.DataFrame(kr).drop(columns="main_auroc")
if len(kr) and "main" in kr and kr.main.notna().all():
    kr.to_csv(f"{P}/results/week2_koh.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False))
print("\ninner-fold choice:", sel)
print("\nmatched AUPRC, 9 main tasks, mean/SD over seeds:\n" + mm.to_string())
print("\nthree tests (9-task mean of matched AUPRC):\n" + tests.to_string(index=False))
print("\nKoh 2026 reference, mean over the 8 tasks it covers:\n" + kr.to_string(index=False))
