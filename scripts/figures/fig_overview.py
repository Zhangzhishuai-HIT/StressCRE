"""Figure 1: overview of the study (data, inputs, StressCRE, evaluation, analyses), with the key numbers.
Drawn with matplotlib shapes and text only, so that the PDF imports into PowerPoint as editable shapes.
Usage: python -I fig_overview.py <project>      -> figures/fig1_overview.pdf/.png
"""
import os, sys
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402

P = sys.argv[1]
fs.setup()
tests = pd.read_csv(f"{P}/results/week2_tests.tsv", sep="\t").set_index("model")
W, H = 174, 84
fig = plt.figure(figsize=(W * fs.MM, H * fs.MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


BOXCOL = {  # border, fill
    "Public RNA-seq": ("#2a78d6", "#eaf2fc"), "Inputs per gene": (fs.TEAL, "#e5f5f1"), "StressCRE": (fs.RED, "#fdece9"),
    "Test design": (fs.OCHRE, "#fbf4dd"), "Elements": (fs.UP, "#fbeee4"), "WGD paralogs": (fs.VIOLET, "#efedf8"),
    "Natural variation": (fs.WILD, "#e9f3e7")}


def box(x, y, w, h, title, lines, edge=fs.LIGHT, lw=0.6, fill=fs.BAND, tsize=7.5):
    if title in BOXCOL:
        edge, fill = BOXCOL[title]
        lw = max(lw, 0.9)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fill, ec=edge, lw=lw))
    ax.text(x + 1.2, y + h - 1.4, title, fontsize=tsize, fontweight="bold", va="top", ha="left", color=fs.INK)
    if lines:
        ax.text(x + 1.2, y + h - 5.8, "\n".join(lines), fontsize=6.3, va="top", ha="left", color=fs.INK2, linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=fs.INK2))


Y1, H1 = 43, 38
box(2, Y1, 38, H1, "Public RNA-seq", ["81 studies, 1,717 runs", "118 stress-versus-control comparisons", "drought 46, salt 39, heat 16,", "cold 10, alkaline 7", "DESeq2 per comparison; vote over", "studies + meta-analysis", "up / down / non-responsive:", "10 tasks (5 stresses × 2 directions)"])
box(45, Y1, 40, H1, "Inputs per gene", [])
ax.text(46.2, Y1 + H1 - 5.8, "3,020-bp window; coding sequence masked", fontsize=6.0, va="top", color=fs.INK2)
# window sketch: two halves of 1,500 bp (16 mm each) and the 20-bp spacer
x0, L, y0 = 47.5, 16.0, Y1 + 21.5
tss = x0 + L * 1000 / 1500
x1 = x0 + L + 1.4
tts = x1 + L * 500 / 1500
ax.add_patch(Rectangle((x0, y0), L, 3.0, fc="#8fd3c3", ec="none"))
ax.add_patch(Rectangle((tss + 1.2, y0), x0 + L - tss - 1.2, 3.0, fc="white", ec=fs.INK2, lw=0.4, hatch="////"))
ax.add_patch(Rectangle((x0 + L + 0.2, y0), 1.0, 3.0, fc="white", ec=fs.MUTED, lw=0.4))
ax.add_patch(Rectangle((x1, y0), L, 3.0, fc="#8fd3c3", ec="none"))
ax.add_patch(Rectangle((x1, y0), tts - x1 - 1.0, 3.0, fc="white", ec=fs.INK2, lw=0.4, hatch="////"))
for xx, lab in ((tss, "TSS"), (tts, "TTS")):
    ax.plot([xx, xx], [y0 - 0.6, y0 + 3.8], color=fs.RED, lw=0.8)
    ax.text(xx, y0 + 4.1, lab, fontsize=6, ha="center", va="bottom", color=fs.RED)
for xx, lab, ha in ((x0, "−1,000", "left"), (x0 + L, "+500", "right"), (x1, "−500", "left"), (x1 + L, "+1,000", "right")):
    ax.text(xx, y0 - 0.8, lab, fontsize=6, va="top", ha=ha, color=fs.INK2)
ax.text(66, y0 - 4.2, "hatched: coding sequence (masked)", fontsize=6, ha="center", va="top", color=fs.MUTED)
ax.text(46.2, Y1 + 11.4, "15 gene features: atlas expression level,\nbreadth and tissue specificity (tau), gene\nstructure, window GC, family size and\nduplication", fontsize=6.3, va="top", color=fs.INK2, linespacing=1.35)
box(90, Y1, 38, H1, "StressCRE", ["Convolutional network on the", "one-hot window (from scratch)", "PlantCAD2-Small, frozen:", "60 × 50-bp bins + attention head", "both with the gene features;", "prediction = mean of the two", "10 outputs, masked multi-task loss"], edge=fs.RED, lw=1.0)
box(133, Y1, 39, H1, "Test design", ["Gene groups: protein families,", "WGD pairs, similar windows;", "5 folds × 3 assignments; chromosome", "hold-out, random split", "Expression-matched AUPRC:", f"StressCRE {tests.loc['cand3', 'main']:.3f}", f"Gene features + k-mers {tests.loc['stack', 'main']:.3f}", f"Gene features {tests.loc['cov', 'main']:.3f}"])
for xa, xb in ((40.4, 44.6), (85.4, 89.6), (128.4, 132.6)):
    arrow(xa, Y1 + H1 / 2, xb, Y1 + H1 / 2)
Y2, H2 = 3, 28
box(2, Y2, 54, H2, "Elements", ["Contribution scores + TF-MoDISco-lite: the ABRE", "(ACGTGG/T) is the main learned element of all five", "up-regulation tasks (3–11% of the positive", "contribution in 0.3–0.4% of the bases)", "Insertion into non-responsive genes: strongest", "near the TSS, both orientations, two ABREs", "more than additive"])
box(60, Y2, 54, H2, "WGD paralogs", ["15,487 pairs; 328 divergent in at least one task", "The responsive copy is the more tissue-", "specific one (61% of divergent combinations)", "Copy-specific windows do not predict", "which copy responds"])
box(118, Y2, 54, H2, "Natural variation", ["1,511 accessions: 3.2 million SNP–gene", "combinations; large predicted effects enriched", "for ABRE gains and losses; alleles that", "destroy an ABRE are more often rare", "Wild soybean W05: salt response predicted", "from its orthologs (AUROC 0.80; tau alone", "0.77; W05 not used in training)"])
for xm in (29, 87, 145):
    arrow(109, Y1 - 0.6, xm, Y2 + H2 + 0.8)
fs.save(fig, f"{P}/figures", "fig1_overview")
print("written figures/fig1_overview")
