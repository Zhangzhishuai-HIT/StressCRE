"""Figure 2: the stress-response labels and how much the genes' baseline expression explains.

A: studies and stress-vs-control comparisons per stress (labels v1).
B: genes per task: strict up, strict down and non-responsive.
C: tissue specificity (tau, JGI expression atlas) of responsive and non-responsive genes per stress.
D: AUROC per task of tau alone, the gene-feature model and the k-mer model (test sets of the main split, mean over the three fold assignments).
Usage: python -I fig_data.py <project>     -> figures/fig2_data.pdf/.png
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle as fs   # noqa: E402
from cre_data import strict_calls   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402

P = sys.argv[1]
fs.setup()
cnt = pd.read_csv(f"{P}/results/label_counts_v1.tsv", sep="\t").set_index("stress").loc[fs.STRESS]
lab = strict_calls(pd.read_csv(f"{P}/labels/labels_v1.tsv.gz", sep="\t"))
cov = pd.read_csv(f"{P}/features/gene_covariates.tsv", sep="\t").set_index("gene")
G1, G2, G3 = fs.UP, fs.DOWN, fs.NONRESP      # up, down, non-responsive
fig = plt.figure(figsize=(fs.W2, 105 * fs.MM))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], hspace=0.62, wspace=0.3, left=0.1, right=0.985, top=0.95, bottom=0.13)

axA = fig.add_subplot(gs[0, 0])
y = np.arange(len(fs.STRESS))[::-1]
axA.barh(y + 0.18, cnt.studies, height=0.34, color=fs.STUDY, label="Studies")
axA.barh(y - 0.18, cnt.contrasts, height=0.34, color=fs.COMPARISON, label="Comparisons")
for yy, a, b in zip(y, cnt.studies, cnt.contrasts):
    axA.text(a + 0.6, yy + 0.18, str(a), va="center", fontsize=6)
    axA.text(b + 0.6, yy - 0.18, str(b), va="center", fontsize=6)
axA.set_yticks(y)
axA.set_yticklabels([fs.STRESS_LABEL[s] for s in fs.STRESS])
axA.set_xlabel("Number")
axA.legend(loc="lower right", handlelength=1, borderaxespad=0.2)
fs.panel(axA, "A", x=-0.22, y=1.03)

axB = fig.add_subplot(gs[0, 1])
x = np.arange(len(fs.STRESS))
up, dn, nr = cnt.strict_up.values, cnt.strict_down.values, cnt.strict_nonresp.values
axB.bar(x - 0.27, up, width=0.26, color=G1, label="Up-regulated")
axB.bar(x, dn, width=0.26, color=G2, label="Down-regulated")
axB.bar(x + 0.27, nr, width=0.26, color=G3, label="Non-responsive")
axB.set_yscale("log")
axB.set_ylim(10, 4e4)
axB.set_xticks(x)
axB.set_xticklabels([fs.STRESS_LABEL[s] for s in fs.STRESS])
axB.set_ylabel("Genes")
axB.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1, columnspacing=1, borderaxespad=0.1)
fs.panel(axB, "B", x=-0.12, y=1.03)

axC = fig.add_subplot(gs[1, 0])
tau = cov.expr_tau
data, pos, cols = [], [], []
for k, s in enumerate(fs.STRESS):
    q = lab[lab.stress == s]
    r = tau.reindex(q[q.call.isin(["up", "down"])].gene).dropna().values
    n = tau.reindex(q[q.call == "nonresp"].gene).dropna().values
    data += [r, n]
    pos += [k * 3, k * 3 + 1]
    cols += [fs.RESP, fs.NONRESP]
bp = axC.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False, whis=(5, 95), medianprops=dict(color="white", lw=0.8),
                 whiskerprops=dict(lw=0.6, color=fs.INK2), capprops=dict(lw=0.6, color=fs.INK2), boxprops=dict(lw=0))
for b, c in zip(bp["boxes"], cols):
    b.set_facecolor(c)
for m in bp["medians"][1::2]:          # dark median line on the light boxes
    m.set_color(fs.INK)
axC.set_xticks([k * 3 + 0.5 for k in range(len(fs.STRESS))])
axC.set_xticklabels([fs.STRESS_LABEL[s] for s in fs.STRESS])
axC.set_ylabel("Tissue specificity (tau)")
axC.legend([bp["boxes"][0], bp["boxes"][1]], ["Responsive", "Non-responsive"], loc="lower left", bbox_to_anchor=(0, 1.0),
           ncol=2, handlelength=1, borderaxespad=0.1)
fs.panel(axC, "C", x=-0.22, y=1.03)

axD = fig.add_subplot(gs[1, 1])
pr = {(m, sd): pd.read_csv(f"{P}/results/v1_{m}_main_seed{sd}_predictions.tsv.gz", sep="\t") for m in ("cov", "kmer") for sd in range(3)}
res = {"tau": [], "cov": [], "kmer": []}
for t in fs.TASKS:                      # model AUROCs: mean over the three fold assignments; tau does not depend on them
    q = pr[("cov", 0)][pr[("cov", 0)].task == t]
    tt = tau.reindex(q.gene).fillna(tau.min()).values
    res["tau"].append(roc_auc_score(q.y, tt))
    for m in ("cov", "kmer"):
        res[m].append(np.mean([roc_auc_score(d.y, d.pred) for d in (pr[(m, sd)][pr[(m, sd)].task == t] for sd in range(3))]))
xs = np.arange(len(fs.TASKS))
for key, (lab_, col, mk) in (("cov", fs.MODEL["cov"]), ("kmer", fs.MODEL["kmer"]), ("tau", ("Tau alone", fs.TAU, "_"))):
    dx = {"cov": -0.17, "kmer": 0.0, "tau": 0.17}[key]          # side by side, so that no marker hides another
    axD.scatter(xs + dx, res[key], marker=mk, s=18 if key != "tau" else 30, color=col, label=lab_, zorder=3,
                linewidth=1.2 if key == "tau" else 0.4, edgecolor="white" if key != "tau" else None)
fs.task_axis(axD, y_group=-0.15)
axD.axhline(0.5, color=fs.MUTED, lw=0.6, ls=(0, (3, 2)))
axD.set_ylim(0.45, 1.0)
axD.set_ylabel("AUROC (test sets)")
axD.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), handletextpad=0.2, columnspacing=1, borderaxespad=0.1)
fs.panel(axD, "D", x=-0.12, y=1.03)
fs.save(fig, f"{P}/figures", "fig2_data")
pd.DataFrame({"task": fs.TASKS, **{k: np.round(v, 3) for k, v in res.items()}}).to_csv(f"{P}/results/fig2_auroc.tsv", sep="\t", index=False)
print("written figures/fig2_data; AUROC tau", np.round(res["tau"], 2), "cov", np.round(res["cov"], 2), "kmer", np.round(res["kmer"], 2))
