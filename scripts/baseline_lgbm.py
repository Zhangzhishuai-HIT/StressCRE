"""LightGBM baselines on a chosen feature set (plan step 5), same protocol as baseline_kmer.py.
FEATURES env var: kmer | covariates | kmer+covariates   (covariates = features/gene_covariates.tsv,
gene-level covariates without promoter sequence content: the gene-feature baseline).

Tasks: 5 stresses x {up, down}; positives = strict up (or down) genes, negatives = non-responsive genes,
"unclear" genes are left out of that task. Features: features/kmer_a4.npy (canonical 1-6-mers, TSS and
TTS halves; no label-dependent feature selection).
For every test fold: early stopping on one inner fold taken from the training folds, then refit on all
training folds with that number of trees, then predict the test fold.

Usage: python3 -I baseline_kmer.py <project_dir> <split> <labels.tsv.gz> <out_prefix> [n_jobs]
  split = splits/fold_seed0.tsv | splits/random_seed0.tsv  (column 'fold')
        | splits/chrom_holdout.tsv                          (columns 'round', 'role'; pooled over rounds)
Writes <out_prefix>_predictions.tsv.gz and <out_prefix>_metrics.tsv.
"""
import os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

P, SPLIT, LAB, OUT = sys.argv[1:5]
NJ = int(sys.argv[5]) if len(sys.argv) > 5 else 8
STRESSES = ["drought", "salt", "heat", "cold", "alkaline"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=20,
              colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=1.0,
              n_jobs=NJ, verbose=-1, random_state=0)

FEATS = os.environ.get("FEATURES", "kmer")
genes = [l.strip() for l in open(f"{P}/features/kmer_a4.genes.txt") if l.strip()]
gi = {g: i for i, g in enumerate(genes)}
parts = []
if "kmer" in FEATS:
    parts.append(np.load(f"{P}/features/kmer_a4.npy"))
if "covariates" in FEATS:
    cov = pd.read_csv(f"{P}/features/gene_covariates.tsv", sep="\t").set_index("gene").reindex(genes)
    parts.append(cov.astype(np.float32).values)
X = np.hstack(parts) if len(parts) > 1 else parts[0]
if FEATS == "covariates":
    PARAMS["colsample_bytree"] = 1.0
print("features", FEATS, X.shape, flush=True)

lab = pd.read_csv(LAB, sep="\t")
lab = lab[lab.stress.isin(STRESSES)].copy()
lab["call"] = "unclear"
lab.loc[(lab.vote == "up") & (lab.meta == "up"), "call"] = "up"
lab.loc[(lab.vote == "down") & (lab.meta == "down"), "call"] = "down"
lab.loc[(lab.meta == "nonresp") & ~lab.vote.isin(["up", "down"]), "call"] = "nonresp"

sp = pd.read_csv(f"{P}/{SPLIT}", sep="\t")
chrom_mode = "role" in sp.columns
if chrom_mode:
    rounds = sorted(sp["round"].unique())
else:
    fold = dict(zip(sp.gene, sp.fold))


def fit_predict(Xtr, ytr, Xva, yva, Xte):
    m = lgb.LGBMClassifier(n_estimators=3000, **PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    best = max(m.best_iteration_ or 100, 20)
    m2 = lgb.LGBMClassifier(n_estimators=best, **PARAMS)
    m2.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    return m2.predict_proba(Xte)[:, 1], best


preds, rows = [], []
t0 = time.time()
for s in STRESSES:
    ls = lab[lab.stress == s].set_index("gene").call
    for d in ("up", "down"):
        task = f"{s}-{d}"
        g = [x for x in ls.index if ls[x] in (d, "nonresp") and x in gi]
        y = np.array([1 if ls[x] == d else 0 for x in g])
        idx = np.array([gi[x] for x in g])
        if chrom_mode:
            units = []
            for r in rounds:
                role = dict(zip(sp.gene[sp["round"] == r], sp.role[sp["round"] == r]))
                tr = np.array([role[x] == "train" for x in g]); te = np.array([role[x] == "test" for x in g])
                units.append((r, tr, te))
        else:
            f = np.array([fold[x] for x in g])
            units = [(k, f != k, f == k) for k in range(5)]
        for k, tr_mask, te_mask in units:
            if te_mask.sum() == 0 or y[te_mask].sum() == 0:
                continue
            tr_idx = np.where(tr_mask)[0]
            # inner validation: one training fold (main/random) or a 20% group-free random slice (chrom)
            if chrom_mode:
                rng = np.random.default_rng(k)
                va_sel = rng.random(len(tr_idx)) < 0.2
            else:
                inner = (k + 1) % 5
                va_sel = np.array([fold[g[i]] == inner for i in tr_idx])
            tr_i, va_i = tr_idx[~va_sel], tr_idx[va_sel]
            te_i = np.where(te_mask)[0]
            p, best = fit_predict(np.asarray(X[idx[tr_i]]), y[tr_i], np.asarray(X[idx[va_i]]), y[va_i],
                                  np.asarray(X[idx[te_i]]))
            yt = y[te_i]
            rows.append((task, k, len(te_i), int(yt.sum()), round(yt.mean(), 5),
                         round(average_precision_score(yt, p), 5), round(roc_auc_score(yt, p), 5), best))
            preds.append(pd.DataFrame({"gene": [g[i] for i in te_i], "task": task, "fold": k, "y": yt, "pred": p}))
            print(f"{task} fold {k}: n {len(te_i)} pos {int(yt.sum())} AUPRC {rows[-1][5]:.4f} "
                  f"(prev {rows[-1][4]:.4f}) AUROC {rows[-1][6]:.4f} trees {best}  [{time.time() - t0:.0f}s]", flush=True)

pr = pd.concat(preds)
pr.to_csv(OUT + "_predictions.tsv.gz", sep="\t", index=False)
met = pd.DataFrame(rows, columns=["task", "fold", "n_test", "n_pos", "prevalence", "auprc", "auroc", "trees"])
pooled = []
for task, q in pr.groupby("task"):
    pooled.append((task, "pooled", len(q), int(q.y.sum()), round(q.y.mean(), 5),
                   round(average_precision_score(q.y, q.pred), 5), round(roc_auc_score(q.y, q.pred), 5), np.nan))
met = pd.concat([met, pd.DataFrame(pooled, columns=met.columns)])
if not chrom_mode and "promoter_similar_in_other_fold" in sp.columns:
    sh = dict(zip(sp.gene, sp.promoter_similar_in_other_fold))
    diag = []
    for task, q in pr.groupby("task"):
        for v in (0, 1):
            qq = q[q.gene.map(sh) == v]
            if qq.y.sum() > 0 and (qq.y == 0).sum() > 0:
                diag.append((task, f"promoter_similar_in_other_fold={v}", len(qq), int(qq.y.sum()),
                             round(qq.y.mean(), 5), round(average_precision_score(qq.y, qq.pred), 5),
                             round(roc_auc_score(qq.y, qq.pred), 5), np.nan))
    met = pd.concat([met, pd.DataFrame(diag, columns=met.columns)])
met.to_csv(OUT + "_metrics.tsv", sep="\t", index=False)
summ = met[met.fold == "pooled"].copy()
summ["lift"] = (summ.auprc / summ.prevalence).round(2)
print(summ.to_string(index=False))
