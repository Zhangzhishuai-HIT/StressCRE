# Adapted 2026-11-16 from 2080/SoyMultiRes/scripts/05_figures/figstyle.py (print sizes, Times New Roman, palette style).
"""Shared figure style for the StressCRE paper: journal print sizes, Times New Roman, one fixed colour per model.

Figures are drawn at their final printed size (double column 174 mm, single column 84 mm) with 7-pt text and 0.6-pt
axes, saved as vector PDF (fonts embedded) and 600-dpi PNG.
Model colours (legend order) passed the dataviz palette validator in adjacent order (lightness band, chroma >= 0.1,
CVD dE >= 13.5, normal-vision dE >= 22); all-pairs separation of red vs magenta is lower, so dot plots also use one
marker shape per model. Wild/cultivated colours passed all checks. Stresses are not colour-coded (axis groups and
text), so that every colour keeps one meaning across the figures.
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

MM = 1 / 25.4
W2 = 174 * MM
W1 = 84 * MM
FONT = "Times New Roman"
for _d in ("~/.local/share/fonts", "~/.fonts"):
    for _f in glob.glob(os.path.join(os.path.expanduser(_d), "**", "[Tt]imes*.[Tt][Tt][Ff]"), recursive=True):
        font_manager.fontManager.addfont(_f)

INK, INK2, MUTED, LIGHT, GRID, BAND = "#222222", "#4a4a4a", "#8c8c8c", "#c8c8c8", "#ebebeb", "#f5f5f3"
RED, NAVY, TEAL, OCHRE, VIOLET, MAGENTA = "#E64B35", "#2F4E9E", "#00A087", "#D9A400", "#7B6FC4", "#C2185B"
WILD, CULT = "#3C8D2F", "#4DBBD5"
# stresses (checked with the dataviz validator on white, all pairs: CVD dE >= 9.7, normal vision >= 15.3; lines also
# carry one marker per stress), direction of the response, and other roles
STRESS_COL = {"drought": "#d97b29", "salt": "#2a78d6", "heat": "#c0392b", "cold": "#1baf7a", "alkaline": "#5e3c99"}
STRESS_MARK = {"drought": "o", "salt": "s", "heat": "^", "cold": "D", "alkaline": "v"}
UP, DOWN, NONRESP, RESP = "#C05A00", "#2F4E9E", "#C9C9C9", "#7B3294"
TAU = "#00594C"                       # tau alone: a dark shade of the gene-feature teal
ALL_ACC = "#34495E"                   # all accessions (wild + cultivated)
PC2NET = "#2a78d6"                    # the PlantCAD2 network of StressCRE (Fig. S1)
STUDY, COMPARISON = "#2a78d6", "#86b6ef"
# model key -> (label, colour, marker)
MODEL = {
    "cand3": ("StressCRE", RED, "o"),
    "stack": ("Gene features + k-mers", NAVY, "s"),
    "cov": ("Gene features", TEAL, "D"),
    "kmer": ("k-mers", OCHRE, "^"),
    "deepcre": ("DeepCRE (soybean)", VIOLET, "v"),
    "koh": ("Koh et al. (2026)", MAGENTA, "P"),
}
ORDER = ["cand3", "stack", "cov", "kmer", "deepcre", "koh"]
STRESS = ["drought", "salt", "heat", "cold", "alkaline"]
STRESS_LABEL = {"drought": "Drought", "salt": "Salt", "heat": "Heat", "cold": "Cold", "alkaline": "Alkaline"}
TASKS = [f"{s}-{d}" for s in STRESS for d in ("up", "down")]
TASK_LABEL = {t: f"{STRESS_LABEL[t.split('-')[0]]} {'up' if t.endswith('up') else 'down'}" for t in TASKS}


def setup():
    plt.rcParams.update({
        "font.family": [FONT, "STIXGeneral", "DejaVu Serif"], "font.size": 7, "axes.titlesize": 7,
        "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 3, "lines.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
        "axes.edgecolor": "#333333", "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "legend.frameon": False, "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold", "mathtext.cal": f"{FONT}:italic", "mathtext.sf": FONT, "mathtext.tt": FONT})


def panel(ax, letter, x=-0.12, y=1.04):
    """Panel label in lower case, as in The Plant Journal's legends ("(a)", "(b)", ...)."""
    ax.text(x, y, letter.lower(), transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name + ".pdf"))
    fig.savefig(os.path.join(out_dir, name + ".png"), dpi=600)
    plt.close(fig)


def task_axis(ax, n_tasks=10, y_group=-0.2, shade=True):
    """x axis for the ten tasks: 'up'/'down' ticks, stress names centred below each pair, alternate stresses shaded."""
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(["up" if t.endswith("up") else "down" for t in TASKS[:n_tasks]])
    for k, s in enumerate(STRESS):
        if shade and k % 2 == 0:
            ax.axvspan(2 * k - 0.5, 2 * k + 1.5, color=BAND, zorder=0, lw=0)
        ax.text(2 * k + 0.5 if 2 * k + 1 < n_tasks else 2 * k, y_group, STRESS_LABEL[s], transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7, color=INK)
    ax.set_xlim(-0.6, n_tasks - 0.4)
