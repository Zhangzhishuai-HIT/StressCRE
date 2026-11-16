"""Known-motif baseline (plan step 5, table 4; the approach of Azodi et al. 2020): random forest on the JASPAR
hit counts per motif and window region (features/motif_jaspar_a4.npy from motif_scan.py).

Tasks, labels and splits as in baseline_kmer.py. No label-dependent feature selection; the forest is fitted on
all training folds of a test fold (main / random split) or round (chromosome hold-out) and predicts it.
Random forest: 500 trees, max_features sqrt, min_samples_leaf 3, seed 0.

Usage: python3 -I baseline_motif.py <project_dir> <split> <labels.tsv.gz> <out_prefix> [n_jobs]
Writes <out_prefix>_predictions.tsv.gz and <out_prefix>_metrics.tsv.
"""
import sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

P, SPLIT, LAB, OUT = sys.argv[1:5]
NJ = int(sys.argv[5]) if len(sys.argv) > 5 else 8
STRESSES = ["drought", "salt", "heat", "cold", "alkaline"]
X = np.load(f"{P}/features/motif_jaspar_a4.npy")
genes = [l.strip() for l in open(f"{P}/features/motif_jaspar_a4.genes.txt") if l.strip()]
gi = {g: i for i, g in enumerate(genes)}
lab = pd.read_csv(LAB, sep="\t")
lab = lab[lab.stress.isin(STRESSES)].copy()
lab["call"] = "unclear"
lab.loc[(lab.vote == "up") & (lab.meta == "up"), "call"] = "up"
lab.loc[(lab.vote == "down") & (lab.meta == "down"), "call"] = "down"
lab.loc[(lab.meta == "nonresp") & ~lab.vote.isin(["up", "down"]), "call"] = "nonresp"
sp = pd.read_csv(f"{P}/{SPLIT}", sep="\t")
chrom = "role" in sp.columns
preds, rows, t0 = [], [], time.time()
for s in STRESSES:
    ls = lab[lab.stress == s].set_index("gene").call
    for d in ("up", "down"):
        task = f"{s}-{d}"
        g = [x for x in ls.index if ls[x] in (d, "nonresp") and x in gi]
        y = np.array([1 if ls[x] == d else 0 for x in g])
        idx = np.array([gi[x] for x in g])
        if chrom:
            units = []
            for r in sorted(sp["round"].unique()):
                role = dict(zip(sp.gene[sp["round"] == r], sp.role[sp["round"] == r]))
                units.append((r, np.array([role[x] == "train" for x in g]), np.array([role[x] == "test" for x in g])))
        else:
            fold = dict(zip(sp.gene, sp.fold))
            f = np.array([fold[x] for x in g])
            units = [(k, f != k, f == k) for k in range(5)]
        for k, tr, te in units:
            if y[te].sum() == 0:
                continue
            m = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=3, n_jobs=NJ, random_state=0)
            m.fit(X[idx[tr]], y[tr])
            p = m.predict_proba(X[idx[te]])[:, 1]
            yt = y[te]
            rows.append((task, k, int(te.sum()), int(yt.sum()), round(yt.mean(), 5),
                         round(average_precision_score(yt, p), 5), round(roc_auc_score(yt, p), 5)))
            preds.append(pd.DataFrame({"gene": np.array(g)[te], "task": task, "fold": k, "y": yt, "pred": p}))
            print(f"{task} fold {k}: AUPRC {rows[-1][5]:.4f} (prev {rows[-1][4]:.4f}) AUROC {rows[-1][6]:.4f}  [{time.time() - t0:.0f}s]", flush=True)
pr = pd.concat(preds)
pr.to_csv(OUT + "_predictions.tsv.gz", sep="\t", index=False)
met = pd.DataFrame(rows, columns=["task", "fold", "n_test", "n_pos", "prevalence", "auprc", "auroc"])
pooled = [(t, "pooled", len(q), int(q.y.sum()), round(q.y.mean(), 5), round(average_precision_score(q.y, q.pred), 5),
           round(roc_auc_score(q.y, q.pred), 5)) for t, q in pr.groupby("task")]
met = pd.concat([met, pd.DataFrame(pooled, columns=met.columns)])
met.to_csv(OUT + "_metrics.tsv", sep="\t", index=False)
s = met[met.fold == "pooled"].copy()
s["lift"] = (s.auprc / s.prevalence).round(2)
print(s.to_string(index=False))
