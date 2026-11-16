"""Figure 5: WGD paralogs, wild soybean W05 and natural variation.

A: divergent WGD pairs (one copy responsive, the other non-responsive; all tasks pooled): paired AUROC (share of pairs
   in which the responsive copy gets the higher prediction) with 95% bootstrap intervals.
B: share of divergent pairs in which the responsive copy is more tissue-specific, expressed in fewer samples or
   expressed at a lower level (sign test).
C: W05 salt response predicted from the W05 window and from the Wm82 window of the same gene (AUPRC, 95% CI of the
   difference in the text).
D-E (when variants/scored_snps.tsv.gz exists): D, StressCRE's change in predicted up-regulation (mean of the five
   up-regulation tasks) for SNPs that destroy or create an ABRE and for other SNPs (scored subset); E, Cochran-Mantel-
   Haenszel odds ratios of being rare (MAF < 5%) for large-effect, ABRE-destroying and ABRE-creating SNPs (all
   accessions, wild, cultivated) and of large effect among differentiated SNPs; filled, StressCRE (300,000 rows);
   open, all rows (effect sizes from the convolutional network; the ABRE tests use no predicted effect).
Usage: python -I fig_evolution.py <project>      -> figures/fig5_evolution.pdf/.png
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402

P = sys.argv[1]
fs.setup()


def pfmt(p):
    if p >= 0.001:
        return f"$P$ = {p:.3f}"
    m, e = f"{p:.0e}".split("e")
    return f"$P$ = {m} × 10$^{{{int(e)}}}$"

VAR = os.path.exists(f"{P}/variants/scored_snps.tsv.gz")
fig = plt.figure(figsize=(fs.W2, (120 if VAR else 60) * fs.MM))
gs = fig.add_gridspec(2 if VAR else 1, 3, hspace=0.6, wspace=0.5, left=0.135, right=0.985, top=0.93 if VAR else 0.86, bottom=0.1 if VAR else 0.2)

axA = fig.add_subplot(gs[0, 0])
po = pd.read_csv(f"{P}/results/paralog_pooled.tsv", sep="\t").set_index("model")
items = [("stresscre_shared", "StressCRE,\nwindows only", fs.RED, "white"), ("stresscre_own", "StressCRE", fs.RED, fs.RED),
         ("cov", "Gene features", fs.TEAL, fs.TEAL), ("kmer", "k-mers", fs.OCHRE, fs.OCHRE)]
for i, (k, lab, col, fc) in enumerate(items):
    r = po.loc[k]
    axA.plot([r.auroc_lo, r.auroc_hi], [i, i], color=col, lw=1.0)
    axA.scatter(r.pair_auroc, i, s=22, color=fc, edgecolor=col, linewidth=0.9, zorder=3)
axA.axvline(0.5, color=fs.MUTED, lw=0.6, ls=(0, (3, 2)))
axA.set_yticks(range(len(items))); axA.set_yticklabels([x[1] for x in items]); axA.invert_yaxis()
axA.set_xlabel("Paired AUROC, divergent combinations")
axA.set_xlim(0.4, 0.75)
fs.panel(axA, "A", x=-0.62, y=1.04)

axB = fig.add_subplot(gs[0, 1])
pp = pd.read_csv(f"{P}/results/paralog_pairs.tsv.gz", sep="\t")
cov = pd.read_csv(f"{P}/features/gene_covariates.tsv", sep="\t").set_index("gene")
dv = pp[pp.divergent]
vals = []
for feat, hi, lab in (("expr_tau", True, "More tissue-\nspecific"), ("expr_breadth", False, "Fewer\natlas runs"),
                      ("expr_mean_log2tpm", False, "Lower\nexpression")):
    a, b = cov[feat].reindex(dv.gene_a).values, cov[feat].reindex(dv.gene_b).values
    resp, non = np.where(dv.resp_sign > 0, a, b), np.where(dv.resp_sign > 0, b, a)
    ok = ~np.isnan(resp) & ~np.isnan(non) & (resp != non)
    k = int(((resp > non) if hi else (resp < non))[ok].sum())
    vals.append((lab, k / ok.sum(), binomtest(k, ok.sum(), 0.5).pvalue, ok.sum()))
axB.bar(range(3), [v[1] for v in vals], width=0.6, color=fs.INK2)
axB.axhline(0.5, color=fs.MUTED, lw=0.6, ls=(0, (3, 2)))
for i, v in enumerate(vals):
    axB.text(i, v[1] + 0.01, pfmt(v[2]), ha="center", va="bottom", fontsize=6)
axB.set_xticks(range(3)); axB.set_xticklabels([v[0] for v in vals])
axB.set_ylim(0.4, 0.68)
axB.set_ylabel("Share of divergent\ncombinations")
fs.panel(axB, "B", x=-0.32, y=1.04)

axC = fig.add_subplot(gs[0, 2])
w = pd.read_csv(f"{P}/results/w05_validation.tsv", sep="\t")
w = w[w.subset == "all orthologs"].set_index("task")
x = np.arange(2)
for j, (col, lab, fc) in enumerate((("auprc_wm82", "Wm82 window", "white"), ("auprc_w05", "W05 window", fs.RED))):
    axC.bar(x + (j - 0.5) * 0.36, [w.loc["salt-up", col], w.loc["salt-down", col]], width=0.34, color=fc, edgecolor=fs.RED, lw=0.8, label=lab)
prev = [w.loc["salt-up", "pos"] / w.loc["salt-up", "n"], w.loc["salt-down", "pos"] / w.loc["salt-down", "n"]]
for xx, p_ in zip(x, prev):
    axC.plot([xx - 0.4, xx + 0.4], [p_, p_], color=fs.MUTED, lw=0.6, ls=(0, (3, 2)))
axC.set_xticks(x); axC.set_xticklabels(["Salt up", "Salt down"])
axC.set_ylabel("AUPRC, W05 salt response")
axC.legend(loc="upper right", handlelength=1, frameon=False)
axC.set_ylim(0, 0.72)
fs.panel(axC, "C", x=-0.32, y=1.04)

if VAR:
    snp = pd.read_csv(f"{P}/variants/scored_snps.tsv.gz", sep="\t", usecols=["abre", "large"] + [f"d_{t}" for t in fs.TASKS if t.endswith("-up")])
    summ = pd.read_csv(f"{P}/results/variant_summary.tsv", sep="\t")
    gd = gs[1, :].subgridspec(1, 2, width_ratios=[1, 1.9], wspace=0.55)
    axD = fig.add_subplot(gd[0, 0])
    up = snp[[c for c in snp.columns if c.startswith("d_")]].mean(1).values * 100
    groups = [("destroyed", "Destroys\nan ABRE"), ("created", "Creates\nan ABRE"), ("none", "Other\nSNPs")]
    data = [up[(snp.abre == k).values] for k, _ in groups]
    bp = axD.boxplot(data, whis=(5, 95), showfliers=False, widths=0.55, patch_artist=True,
                     medianprops=dict(color=fs.INK, lw=0.9), whiskerprops=dict(color=fs.INK2, lw=0.7),
                     capprops=dict(color=fs.INK2, lw=0.7), boxprops=dict(edgecolor=fs.INK2, lw=0.7))
    for patch, col in zip(bp["boxes"], ("#2F4E9E", "#C05A00", "#DDDDDD")):   # direction colours of Fig. 4
        patch.set_facecolor(col)
    axD.axhline(0, color=fs.MUTED, lw=0.6, ls=(0, (3, 2)))
    axD.set_xticks(range(1, 4)); axD.set_xticklabels([g[1] for g in groups])
    for i, d in enumerate(data, 1):
        axD.text(i, 1.0, f"n = {len(d):,}", transform=axD.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.5, color=fs.MUTED)
    axD.set_ylabel("Change in predicted up-regulation\n(percentage points, mean of five tasks)")
    fs.panel(axD, "D", x=-0.42, y=1.08)
    axE = fig.add_subplot(gd[0, 1])
    rowsE = [("large effect -> rare (MAF<5%)", "Large effect: rare"), ("ABRE destroyed -> rare (MAF<5%)", "Destroys an ABRE: rare"),
             ("ABRE created -> rare (MAF<5%)", "Creates an ABRE: rare")]
    grp = [("all accessions", "All", fs.INK2), ("wild", "Wild", fs.WILD), ("cultivated", "Cultivated", fs.CULT)]
    y, ticks, labels = 0, [], []
    for key, lab in rowsE:
        for g, gl, col in grp:
            mods = (("StressCRE", col, -0.15), ("CNN, all rows", "white", 0.15)) if key.startswith("large") else (("CNN, all rows", "white", 0.0),)
            for mdl, fc, dy in mods:              # ABRE tests use no predicted effect: all combinations only
                r = summ[(summ.model == mdl) & (summ.analysis == f"{key}, {g}")]
                if len(r):
                    r = r.iloc[0]
                    axE.plot([r.ci_lo, r.ci_hi], [y + dy, y + dy], color=col, lw=0.9)
                    axE.scatter(r.odds_ratio, y + dy, s=16, color=fc, edgecolor=col, linewidth=0.8, zorder=3)
            ticks.append(y); labels.append(f"{lab}, {gl.lower()}" if g != "all accessions" else lab)
            y += 1
        y += 0.5
    r0 = "differentiated (|AF cult - AF wild| >= 0.5) -> large effect, common SNPs"
    for mdl, fc, dy in (("StressCRE", fs.INK2, -0.15), ("CNN, all rows", "white", 0.15)):
        r = summ[(summ.model == mdl) & (summ.analysis == r0)]
        if len(r):
            r = r.iloc[0]
            axE.plot([r.ci_lo, r.ci_hi], [y + dy, y + dy], color=fs.INK2, lw=0.9)
            axE.scatter(r.odds_ratio, y + dy, s=16, color=fc, edgecolor=fs.INK2, linewidth=0.8, zorder=3)
    ticks.append(y); labels.append("Differentiated: large effect")
    axE.axvline(1, color=fs.MUTED, lw=0.6, ls=(0, (3, 2)))
    axE.set_yticks(ticks); axE.set_yticklabels(labels, fontsize=6); axE.invert_yaxis()
    axE.set_xlabel("Odds ratio (Cochran–Mantel–Haenszel, 95% CI)")
    axE.scatter([], [], s=16, color=fs.INK2, label="StressCRE, 300,000 scored combinations")
    axE.scatter([], [], s=16, color="white", edgecolor=fs.INK2, linewidth=0.8, label="All combinations")
    axE.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=5.5, frameon=False, handletextpad=0.2, columnspacing=1.0)
    fs.panel(axE, "E", x=-0.55, y=1.1)
fs.save(fig, f"{P}/figures", "fig5_evolution")
print("written figures/fig5_evolution", "with variant panels" if VAR else "(no variant panels yet)")
