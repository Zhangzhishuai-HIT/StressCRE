"""Cross-genome test with wild soybean W05 (plan step 8): does the W05 window predict the W05 salt response better
than the Wm82 window of the same gene, where the two windows differ?

W05 salt response: DESeq2 on the W05 salt study PRJNA521577 (leaf and root, 15 salt-treated vs 3 control samples
each; reads quantified on the Wm82.a4 gene models by SoyExpFM; work/de_w05). Per organ, up = padj < 0.05 and
log2FC > 1, down = padj < 0.05 and log2FC < -1, non-responsive = not (padj < 0.05 and |log2FC| > 0.5); a gene is
W05 salt-up if up in at least one organ and down in none, non-responsive if non-responsive in both (salt-down
symmetric). W05 data never entered the training labels.
Predictions: results/w05_predictions.tsv.gz (W05 window, Wm82 gene features) and results/v1_stresscre_allgenes.tsv.gz
(Wm82 window, same features); orthologs and window edit distance: seqs/w05_homwindows.tsv (w05_homwindows.py).
Tests, on all orthologs and on those whose windows differ by >= 2% edit distance:
  (1) AUROC and AUPRC of each window's prediction for the W05 labels, paired bootstrap over genes (1,000) for the
      difference W05 window - Wm82 window;
  (2) Spearman correlation between the predicted difference (W05 - Wm82 window) and the measured difference in the
      salt log2 fold change (W05 mean of leaf and root - Wm82 meta-analysis log2 fold change).
Usage: python -I w05_analysis.py <project>    -> results/w05_validation.tsv
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_data import strict_calls   # noqa: E402

P = sys.argv[1]
rng = np.random.default_rng(0)
de = {}
for c, organ in (("salt__PRJNA521577__002", "leaf"), ("salt__PRJNA521577__003", "root")):
    d = pd.read_csv(f"{P}/work/de_w05/{c}/de.tsv.gz", sep="\t").dropna(subset=["log2FoldChange"]).set_index("gene")
    padj = d.padj.fillna(1)
    lfc = d.log2FoldChange
    de[organ] = pd.DataFrame({"lfc": lfc,
                              "call": np.where((padj < 0.05) & (lfc > 1), "up", np.where((padj < 0.05) & (lfc < -1), "down",
                                       np.where(~((padj < 0.05) & (lfc.abs() > 0.5)), "nonresp", "other")))})
w = de["leaf"].join(de["root"], lsuffix="_leaf", rsuffix="_root", how="inner")
w["lfc_w05"] = (w.lfc_leaf + w.lfc_root) / 2
lab = {}
for d in ("up", "down"):
    o = "down" if d == "up" else "up"
    y = pd.Series(np.nan, index=w.index)
    pos = ((w.call_leaf == d) | (w.call_root == d)) & (w.call_leaf != o) & (w.call_root != o)
    neg = (w.call_leaf == "nonresp") & (w.call_root == "nonresp")
    y[pos] = 1; y[neg] = 0
    lab[d] = y
orth = pd.read_csv(f"{P}/seqs/w05_homwindows.tsv", sep="\t")      # homologous-anchor windows and their edit distance
pw = pd.read_csv(f"{P}/results/w05_predictions.tsv.gz", sep="\t")
pa = pd.read_csv(f"{P}/results/v1_stresscre_allgenes.tsv.gz", sep="\t")
pa = pa[pa.task.isin(["salt-up", "salt-down"])]
v1 = strict_calls(pd.read_csv(f"{P}/labels/labels_v1.tsv.gz", sep="\t"))
lfc_wm82 = v1[v1.stress == "salt"].set_index("gene").meta_lfc
rows = []
for d in ("up", "down"):
    t = f"salt-{d}"
    q = orth.merge(pw[pw.task == t][["gene_a4", "p_w05"]], on="gene_a4").merge(
        pa[pa.task == t][["gene", "p"]].rename(columns={"gene": "gene_a4", "p": "p_wm82"}), on="gene_a4")
    q["y"] = lab[d].reindex(q.gene_a4).values
    q["lfc_w05"] = w.lfc_w05.reindex(q.gene_a4).values
    q["lfc_wm82"] = lfc_wm82.reindex(q.gene_a4).values
    for subset, m in (("all orthologs", np.ones(len(q), bool)), ("windows differ >= 2%", q.window_edit_frac.values >= 0.02)):
        s = q[m & q.y.notna().values]
        y = s.y.values.astype(int)
        res = dict(task=t, subset=subset, n=len(s), pos=int(y.sum()))
        if y.sum() >= 5 and (1 - y).sum() >= 5:
            for k in ("p_w05", "p_wm82"):
                res[f"auroc_{k[2:]}"] = round(roc_auc_score(y, s[k]), 4)
                res[f"auprc_{k[2:]}"] = round(average_precision_score(y, s[k]), 4)
            da, dp = [], []
            for _ in range(1000):
                ii = rng.integers(0, len(s), len(s))
                yy = y[ii]
                if yy.sum() == 0 or yy.sum() == len(yy):
                    continue
                da.append(roc_auc_score(yy, s.p_w05.values[ii]) - roc_auc_score(yy, s.p_wm82.values[ii]))
                dp.append(average_precision_score(yy, s.p_w05.values[ii]) - average_precision_score(yy, s.p_wm82.values[ii]))
            res.update(auroc_diff=round(res["auroc_w05"] - res["auroc_wm82"], 4), auroc_diff_lo=round(np.percentile(da, 2.5), 4),
                       auroc_diff_hi=round(np.percentile(da, 97.5), 4), auprc_diff=round(res["auprc_w05"] - res["auprc_wm82"], 4),
                       auprc_diff_lo=round(np.percentile(dp, 2.5), 4), auprc_diff_hi=round(np.percentile(dp, 97.5), 4))
        s2 = q[m].dropna(subset=["lfc_w05", "lfc_wm82"])
        sign = 1 if d == "up" else -1
        r = spearmanr(s2.p_w05 - s2.p_wm82, sign * (s2.lfc_w05 - s2.lfc_wm82))
        res.update(n_lfc=len(s2), spearman_diff=round(r.correlation, 4), spearman_p=float(f"{r.pvalue:.3g}"))
        rows.append(res)
out = pd.DataFrame(rows)
out.to_csv(f"{P}/results/w05_validation.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False))
