"""Figure 4: the elements StressCRE uses.

A: contribution weight matrix of the motif with the largest summed contribution of its sites in each up-regulation
   task (TF-MoDISco-lite on the contribution scores of the convolutional network; an ABRE-like motif in all five), with
   its number of sites.
B: mean net contribution of exact ABRE sites (ACGTGG/T, either strand) in the windows of up-regulated genes, per
   100-bp bin of the TSS half and the TTS half (results/abre_contrib_position.tsv from abre_contrib.py).
C: enrichment of known elements within 500 bp of the TSS among responsive genes (log2 odds ratio; data); dots,
   Benjamini-Hochberg q < 0.05 over the 70 tests.
D: change in StressCRE's log odds when one element is written 150 bp upstream of the TSS of non-responsive genes (98 of
   the 100 genes; constructs overlapping masked bases skipped).
E: the same for the ABRE at different positions (TSS: upstream offsets; TTS: downstream offsets).
Usage: python -I fig_motifs.py <project>      -> figures/fig4_elements.pdf/.png
"""
import os, sys
import h5py
import numpy as np
import pandas as pd
import logomaker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402

P = sys.argv[1]
fs.setup()
pat = pd.read_csv(f"{P}/results/modisco_patterns.tsv", sep="\t")
prof = pd.read_csv(f"{P}/results/abre_contrib_position.tsv", sep="\t")
enr = pd.read_csv(f"{P}/results/element_enrichment.tsv", sep="\t")
ins = pd.read_csv(f"{P}/results/insertion_effects.tsv.gz", sep="\t")
UP = ["drought-up", "salt-up", "heat-up", "cold-up", "alkaline-up"]
DIV = LinearSegmentedColormap.from_list("div", ["#1b2f63", fs.NAVY, "#f4f4f4", "#C05A00", "#7a3a00"])
LOGO_COL = {"A": "#00A087", "C": "#2F4E9E", "G": "#D9A400", "T": "#E64B35"}
fig = plt.figure(figsize=(fs.W2, 175 * fs.MM))
outer = fig.add_gridspec(3, 1, height_ratios=[0.5, 1.0, 1.05], hspace=0.45, left=0.1, right=0.93, top=0.95, bottom=0.08)
T9 = fs.TASKS[:-1]          # alkaline down (32 genes, secondary task) not shown

# A: logos
ga = outer[0].subgridspec(1, 5, wspace=0.25)
tops = {}
for j, t in enumerate(UP):
    q = pat[pat.task == t].assign(total=lambda d: d.n_seqlets * d.cwm_sum).sort_values("total", ascending=False)
    r = q.iloc[0]                                        # motif with the largest summed contribution of its sites
    assert ("ACGTG" in r.consensus.upper() or "CACGT" in r.consensus.upper()), (t, r.consensus)
    tops[t] = r["name"]
    cwm = h5py.File(f"{P}/work/modisco/{t}/modisco.h5", "r")[f"{r.metacluster}_patterns"][f"pattern_{r.pattern}"]["contrib_scores"][:]
    sc = np.abs(cwm).sum(1)
    c = int(np.convolve(sc, np.ones(12), "valid").argmax())
    ax = fig.add_subplot(ga[0, j])
    logomaker.Logo(pd.DataFrame(cwm[c:c + 12], columns=list("ACGT")), ax=ax, color_scheme=LOGO_COL, show_spines=False, baseline_width=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{fs.TASK_LABEL[t]}\n{r.n_seqlets:,} sites", fontsize=6.5, pad=2)
    if j == 0:
        fs.panel(ax, "A", x=-0.45, y=1.12)

# B: positions; C: enrichment
gb = outer[1].subgridspec(1, 3, width_ratios=[1, 1, 1.6], wspace=0.45)
marks = ["o", "s", "^", "D", "v"]
axb0 = None
for k, (half, lab) in enumerate([("TSS", "Position relative to TSS (bp)"), ("TTS", "Position relative to TTS (bp)")]):
    ax = fig.add_subplot(gb[0, k], sharey=axb0)
    axb0 = axb0 or ax
    for j, tk in enumerate(UP):
        q = prof[(prof.task == tk) & (prof.half == half)].sort_values("bin_start")
        ax.plot(q.bin_start + 50, q.mean_net_contribution, marker=marks[j], ms=2.5, lw=0.8, color=fs.STRESS_COL[tk.split("-")[0]],
                label=fs.STRESS_LABEL[tk.split("-")[0]])
    ax.axvline(0, color=fs.MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.axhline(0, color=fs.MUTED, lw=0.4)
    ax.set_xlabel(lab)
    ax.set_xticks([-1000, -500, 0, 500] if half == "TSS" else [-500, 0, 500, 1000])
    if k == 0:
        ax.set_ylabel("Contribution per ABRE site")
        fs.panel(ax, "B", x=-0.42, y=1.04)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.legend(fontsize=6, handlelength=1.4, loc="upper right", frameon=False)
axC = fig.add_subplot(gb[0, 2])
EL = ["ABRE", "G-box", "HSE", "DRE/CRT", "GCC-box", "W-box", "EE"]
m = enr.pivot(index="element", columns="task", values="odds_ratio").reindex(index=EL, columns=T9)
lm = np.log2(m.clip(lower=0.125))
im = axC.imshow(lm.values, cmap=DIV, norm=TwoSlopeNorm(0, -2, 2), aspect="auto")
qv = enr.pivot(index="element", columns="task", values="q").reindex(index=EL, columns=T9)
yy, xx = np.where(qv.values < 0.05)
axC.scatter(xx, yy, s=4, color="white", edgecolor=fs.INK, linewidth=0.4, zorder=3)
axC.set_yticks(range(len(EL))); axC.set_yticklabels(EL)
axC.set_xticks(range(len(T9))); axC.set_xticklabels(["up" if t.endswith("up") else "down" for t in T9], rotation=90)
for k, s in enumerate(fs.STRESS):
    axC.text(2 * k + (0.5 if s != "alkaline" else 0), -0.75, fs.STRESS_LABEL[s], ha="center", va="bottom", fontsize=6.5)
for sp in axC.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=axC, fraction=0.04, pad=0.02)
cb.set_label("log2 odds ratio", fontsize=6.5); cb.ax.tick_params(labelsize=6)
fs.panel(axC, "C", x=-0.22, y=1.1)

# D: insertion heatmap; E: ABRE positions
gd = outer[2].subgridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)
axD = fig.add_subplot(gd[0, 0])
ELI = ["ABRE", "G-box", "HSE", "NACRS", "MYC", "DRE/CRT", "GCC-box", "EE", "W-box", "MYB"]
s = ins[ins.partner.isna() & (ins.anchor == "tss") & (ins.offset == -150) & (ins.strand == "+")]
t = s.groupby(["element", "task"]).lor.mean().unstack().reindex(index=ELI, columns=T9)
im2 = axD.imshow(t.values, cmap=DIV, norm=TwoSlopeNorm(0, -0.3, 0.3), aspect="auto")
axD.set_yticks(range(len(ELI))); axD.set_yticklabels(ELI)
axD.set_xticks(range(len(T9))); axD.set_xticklabels(["up" if t_.endswith("up") else "down" for t_ in T9], rotation=90)
for k, s_ in enumerate(fs.STRESS):
    axD.text(2 * k + (0.5 if s_ != "alkaline" else 0), -0.75, fs.STRESS_LABEL[s_], ha="center", va="bottom", fontsize=6.5)
for sp in axD.spines.values():
    sp.set_visible(False)
cb2 = fig.colorbar(im2, ax=axD, fraction=0.035, pad=0.02)
cb2.set_label("Change in log odds", fontsize=6.5); cb2.ax.tick_params(labelsize=6)
fs.panel(axD, "D", x=-0.17, y=1.08)
axE = fig.add_subplot(gd[0, 1])
a = ins[(ins.element == "ABRE") & ins.partner.isna() & ins.task.isin(UP) & (ins.strand == "+")]
pos_lab = [("tss", o) for o in (-900, -700, -500, -350, -250, -150, -100, -50)] + [("tts", 200), ("tts", 500)]
xs = np.arange(len(pos_lab))
for k, tk in enumerate(UP):
    y = [a[(a.task == tk) & (a.anchor == an) & (a.offset == o)].lor.mean() for an, o in pos_lab]
    col = fs.STRESS_COL[tk.split("-")[0]]
    axE.plot(xs[:8], y[:8], marker=marks[k], ms=3, lw=0.8, color=col, label=fs.STRESS_LABEL[tk.split("-")[0]])
    axE.plot(xs[8:], y[8:], marker=marks[k], ms=3, lw=0.8, color=col)
axE.set_xticks(xs)
axE.set_xticklabels([str(o) for _, o in pos_lab[:8]] + [f"+{o}" for _, o in pos_lab[8:]], rotation=90)
axE.axvline(7.5, color=fs.MUTED, lw=0.6, ls=(0, (3, 2)))
axE.set_xlabel("Upstream of TSS (bp)              Downstream of TTS (bp)", fontsize=6.5)
axE.set_ylabel("Change in log odds (ABRE)")
axE.legend(fontsize=6, handlelength=1.6, loc="upper left")
fs.panel(axE, "E", x=-0.3, y=1.08)
fs.save(fig, f"{P}/figures", "fig4_elements")
print("written figures/fig4_elements")
