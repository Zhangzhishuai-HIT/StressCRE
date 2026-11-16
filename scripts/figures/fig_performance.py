"""Figure 3: prediction accuracy of StressCRE and the comparison models (test sets, expression-matched AUPRC).

A: fold enrichment (matched AUPRC / matched prevalence) per task, mean and range over the three fold assignments.
B: mean matched AUPRC over the nine main tasks in the three test designs.
C: StressCRE minus gene features + k-mers, matched AUPRC difference with 95% bootstrap CI, one point per seed.
Usage: python -I fig_performance.py <project>     -> figures/fig3_performance.pdf/.png
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402

P = sys.argv[1]
fs.setup()
te = pd.read_csv(f"{P}/results/v1_models_test_main.tsv", sep="\t")
te["lift"] = te.auprc_matched / te.prevalence_matched
tests = pd.read_csv(f"{P}/results/week2_tests.tsv", sep="\t").set_index("model")
fig = plt.figure(figsize=(fs.W2, 128 * fs.MM))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1, 1.35], hspace=0.5, wspace=0.3,
                      left=0.09, right=0.985, top=0.93, bottom=0.13)
axA = fig.add_subplot(gs[0, :])
x = np.arange(len(fs.TASKS))
off = np.linspace(-0.3, 0.3, len(fs.ORDER))
for k, m in enumerate(fs.ORDER):
    lab, col, mk = fs.MODEL[m]
    q = te[te.model == m].groupby("task").lift
    mean, lo, hi = q.mean(), q.min(), q.max()
    xs, ys, ls, hs = [], [], [], []
    for i, t in enumerate(fs.TASKS):
        if t in mean.index:
            xs.append(i + off[k]); ys.append(mean[t]); ls.append(lo[t]); hs.append(hi[t])
    xs, ys = np.array(xs), np.array(ys)
    axA.vlines(xs, ls, hs, color=col, lw=0.8, zorder=2)
    axA.scatter(xs, ys, s=16 if m == "cand3" else 11, marker=mk, color=col, edgecolor="white", linewidth=0.4,
                zorder=3, label=lab)
fs.task_axis(axA, y_group=-0.11)
axA.axhline(1, color=fs.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
axA.set_ylabel("Fold enrichment\n(matched AUPRC / share)")
axA.legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0), handletextpad=0.2, columnspacing=1.0, borderaxespad=0.2)
fs.panel(axA, "A", x=-0.065, y=1.08)

axB = fig.add_subplot(gs[1, 0])
designs = [("main", "Gene groups\n(5 folds, 3 assignments)"), ("chrom", "Chromosome\nhold-out"), ("random", "Random\ngenes")]
mods = [m for m in fs.ORDER if m in tests.index]
w = 0.8 / len(mods)
for k, m in enumerate(mods):
    lab, col, mk = fs.MODEL[m]
    vals = [tests.loc[m, d] for d, _ in designs]
    xs = np.arange(len(designs)) - 0.4 + w * (k + 0.5)
    axB.bar(xs, vals, width=w * 0.86, color=col, edgecolor="none", zorder=2)
    if m == "cand3":
        for xx, v in zip(xs, vals):
            axB.text(xx, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=6, color=fs.INK)
axB.set_xticks(np.arange(len(designs)))
axB.set_xticklabels([d[1] for d in designs])
axB.set_ylabel("Matched AUPRC\n(mean of nine tasks)")
axB.grid(axis="y", color=fs.GRID, lw=0.5, zorder=0)
axB.set_axisbelow(True)
fs.panel(axB, "B", x=-0.2, y=1.04)

axC = fig.add_subplot(gs[1, 1])
ev = [pd.read_csv(f"{P}/results/eval_v1_cand3_vs_stack_seed{s}.tsv", sep="\t").set_index("task") for s in range(3)]
for i, t in enumerate(fs.TASKS):
    for s in range(3):
        r = ev[s].loc[t]
        sig = r.verdict == "A better"
        xx = i + (s - 1) * 0.18
        axC.vlines(xx, r.diff_lo, r.diff_hi, color=fs.RED if sig else fs.MUTED, lw=0.8)
        axC.scatter(xx, r["diff"], s=9, color=fs.RED if sig else "white", edgecolor=fs.RED if sig else fs.MUTED,
                    linewidth=0.6, zorder=3)
axC.axhline(0, color=fs.INK2, lw=0.6)
fs.task_axis(axC, y_group=-0.13)
axC.set_ylabel("Matched AUPRC difference\n(StressCRE \u2212 gene features + k-mers)")
fs.panel(axC, "C", x=-0.14, y=1.04)
fs.save(fig, f"{P}/figures", "fig3_performance")
print("written figures/fig3_performance")
