"""Supplementary figures.

S1: validation-set expression-matched AUPRC (mean of the nine main tasks) per epoch for the two networks of StressCRE
    (main design, 3 fold assignments x 5 test folds; models/v1_<cnn|pc2>_main_seed*/run.json).
S2: fold enrichment per task in the chromosome hold-out design (StressCRE and comparison models).
S3: WGD pairs per task: paired AUROC on divergent pairs, StressCRE with shared gene features (windows only) and
    the k-mer model, with 95% bootstrap intervals.
Usage: python -I fig_supp.py <project>     -> figures/figS1_training, figS2_chromosome, figS3_paralogs (.pdf/.png)
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle as fs   # noqa: E402
from cre_eval import scores   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402

P = sys.argv[1]
fs.setup()
# S1
fig, axes = plt.subplots(1, 2, figsize=(fs.W2, 60 * fs.MM), sharey=True)
for ax, (kind, lab) in zip(axes, (("cnn", "Convolutional network"), ("pc2", "PlantCAD2 network"))):
    for s in range(3):
        run = json.load(open(f"{P}/models/v1_{kind}_main_seed{s}/run.json"))
        for k, f in run["folds"].items():
            h = pd.DataFrame(f["history"])
            ax.plot(h.epoch, h.val_matched_auprc, color=fs.RED if kind == "cnn" else fs.INK2, lw=0.5, alpha=0.6)
            ax.scatter(f["best_epoch"], h.val_matched_auprc[h.epoch == f["best_epoch"]].values[0], s=6, color=fs.INK, zorder=3)
    ax.axvline(20, color=fs.MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_title(lab, fontsize=7)
    ax.set_xlabel("Epoch")
axes[0].set_ylabel("Validation matched AUPRC\n(mean of nine tasks)")
fs.panel(axes[0], "A", x=-0.2); fs.panel(axes[1], "B", x=-0.08)
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.2, top=0.88, wspace=0.14)
fs.save(fig, f"{P}/figures", "figS1_training")

# S2
cov = pd.read_csv(f"{P}/features/gene_covariates.tsv", sep="\t").set_index("gene")
fig, ax = plt.subplots(figsize=(fs.W2, 62 * fs.MM))
off = np.linspace(-0.3, 0.3, 6)
for k, m in enumerate(fs.ORDER):
    f = f"{P}/results/v1_{m}_chrom_predictions.tsv.gz"
    if not os.path.exists(f):
        continue
    sc = scores(cov, pd.read_csv(f, sep="\t"))
    xs = [i + off[k] for i, t in enumerate(fs.TASKS) if t in sc]
    ys = [sc[t]["auprc_matched"] / sc[t]["prevalence_matched"] for t in fs.TASKS if t in sc]
    lab, col, mk = fs.MODEL[m]
    ax.scatter(xs, ys, s=14 if m == "cand3" else 10, marker=mk, color=col, edgecolor="white", linewidth=0.4, label=lab, zorder=3)
fs.task_axis(ax, y_group=-0.13)
ax.axhline(1, color=fs.MUTED, lw=0.6, ls=(0, (3, 2)))
ax.set_ylabel("Fold enrichment\n(matched AUPRC / share)")
ax.legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0), handletextpad=0.2, columnspacing=0.9)
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.2, top=0.86)
fs.save(fig, f"{P}/figures", "figS2_chromosome")

# S3
pm = pd.read_csv(f"{P}/results/paralog_metrics.tsv", sep="\t")
fig, ax = plt.subplots(figsize=(fs.W2, 62 * fs.MM))
for k, (m, lab, col, fc) in enumerate((("stresscre_shared", "StressCRE, windows only", fs.RED, "white"), ("kmer", "k-mers", fs.OCHRE, fs.OCHRE))):
    q = pm[pm.model == m].set_index("task").reindex(fs.TASKS)
    xs = np.arange(len(fs.TASKS)) + (k - 0.5) * 0.3
    ax.vlines(xs, q.auroc_lo, q.auroc_hi, color=col, lw=0.8)
    ax.scatter(xs, q.pair_auroc, s=14, color=fc, edgecolor=col, linewidth=0.8, zorder=3, label=lab)
for i, t in enumerate(fs.TASKS):
    n = int(pm[(pm.model == "kmer") & (pm.task == t)].n_divergent.iloc[0])
    ax.text(i, 0.02, f"n = {n}", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.5, color=fs.MUTED)
fs.task_axis(ax, y_group=-0.13)
ax.axhline(0.5, color=fs.MUTED, lw=0.6, ls=(0, (3, 2)))
ax.set_ylabel("Paired AUROC,\ndivergent pairs")
ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.2, top=0.86)
fs.save(fig, f"{P}/figures", "figS3_paralogs")
print("written figS1_training, figS2_chromosome, figS3_paralogs")
