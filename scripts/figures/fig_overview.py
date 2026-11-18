"""Figure 1: overview of the study drawn as an architecture diagram.

Top left, the inputs of every gene (sequence window and gene features); top right, the stress-response labels from
public RNA-seq; middle, the two networks of StressCRE and the mean of their predicted probabilities; bottom, the test
design with the main accuracy result and the three analyses that use StressCRE. Key numbers are read from the result
files where one holds them (accuracy, comparisons per stress, paralog pairs, ABRE contribution) and checked against the
values written on the figure.
Drawn with matplotlib shapes and text only, so that the PDF imports into PowerPoint as editable shapes.
Usage: python -I fig_overview.py <project>      -> figures/fig1_overview.pdf/.png
"""
import os, sys
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle, PathPatch, Ellipse, Arc
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
from matplotlib.colors import to_rgb, to_hex

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402

P = sys.argv[1]
fs.setup()

# ---------------------------------------------------------------- numbers written on the figure, checked against results
tests = pd.read_csv(f"{P}/results/week2_tests.tsv", sep="\t").set_index("model")
cnt = pd.read_csv(f"{P}/results/label_counts_v1.tsv", sep="\t").set_index("stress").loc[fs.STRESS]
par = pd.read_csv(f"{P}/results/paralog_pooled.tsv", sep="\t").set_index("model")
el = pd.read_csv(f"{P}/results/element_contributions.tsv", sep="\t")
abre_up = el[(el.element == "ABRE") & el.task.str.endswith("-up")]
ACC = {m: round(float(tests.loc[m, "main"]), 3) for m in ("cand3", "stack", "cov", "kmer")}
assert ACC == {"cand3": 0.155, "stack": 0.128, "cov": 0.119, "kmer": 0.092}, ACC
assert list(cnt.contrasts) == [46, 39, 16, 10, 7] and int(cnt.contrasts.sum()) == 118
assert round(float(par.loc["stresscre_shared", "pair_auroc"]), 2) == 0.54 and int(par.loc["stresscre_shared", "n_divergent"]) == 335
assert (round(abre_up.positive_contribution_share.min() * 100), round(abre_up.positive_contribution_share.max() * 100)) == (3, 11)
assert (round(abre_up.base_share.min() * 100, 1), round(abre_up.base_share.max() * 100, 1)) == (0.3, 0.4)

# ---------------------------------------------------------------- canvas (mm) and helpers
W, H, Y0 = 174, 143, -7.0                  # the canvas runs from y = Y0 to y = H + Y0
fig = plt.figure(figsize=(W * fs.MM, H * fs.MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(Y0, H + Y0); ax.axis("off")
RND = fig.canvas.get_renderer()
rng = np.random.default_rng(7)

BASE = {"A": "#00A087", "C": "#2F4E9E", "G": "#D9A400", "T": "#E64B35"}     # as the motif logos of Fig. 4a
CNN_C, CNN_D = fs.RED, "#B8392A"            # convolutional network (as in Fig. S1) and a darker shade for its text
PC2_C, PC2_D = fs.PC2NET, "#1F5FAF"         # PlantCAD2 network (as in Fig. S1)
GF_C = fs.TEAL                              # gene features (as the gene-feature model)
ARROW = "#59626e"
FLOW = "#b7bfc9"                            # broad arrows between the parts of the figure
MUTE = "#6b6b6b"


def mix(c, other, f):
    a, b = np.array(to_rgb(c)), np.array(to_rgb(other))
    return to_hex(a * (1 - f) + b * f)


def T(x, y, s, size=6, color=fs.INK, ha="left", va="center", weight="normal", style="normal", z=6, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, fontstyle=style, zorder=z, **kw)


def extent(t):
    bb = t.get_window_extent(renderer=RND)
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(bb)
    return x1 - x0, y1 - y0


def wrap(text, width, size=6):
    """Greedy line breaking at spaces by measured text width (mm); non-breaking spaces keep words together."""
    lines, cur = [], ""
    for w in text.split(" "):
        trial = (cur + " " + w).strip()
        t = ax.text(0, 0, trial, fontsize=size)
        tw = extent(t)[0]
        t.remove()
        if tw <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def para(x, ytop, text, width=36.0, step=2.75, size=6, color=None):
    """A left-aligned paragraph; returns the y below its last line."""
    lines = wrap(text, width, size)
    for k, line in enumerate(lines):
        T(x, ytop - k * step, line, va="top", size=size, color=color or fs.INK)
    return ytop - len(lines) * step


def rbox(x, y, w, h, fc, ec="none", lw=0.5, r=1.0, z=2, ls="-", hatch=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z,
                       linestyle=ls, hatch=hatch)
    ax.add_patch(p)
    return p


def rect(x, y, w, h, fc, ec="none", lw=0.3, z=3, hatch=None):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, hatch=hatch))


def arrow(x0, y0, x1, y1, col=ARROW, lw=0.7, ms=5.5, z=4, cs="arc3"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=col, zorder=z,
                                 connectionstyle=cs, shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt"))


def flow(x0, y0, x1, y1, z=1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="simple,head_length=0.9,head_width=1.1,tail_width=0.5",
                                 mutation_scale=7, fc=FLOW, ec="none", zorder=z, shrinkA=0, shrinkB=0))


def region(x, y, w, h, title, col, tint):
    rbox(x, y, w, h, fc=tint, ec=mix(col, "white", 0.35), lw=0.8, r=2.2, z=0)
    t = T(x + 2.7, y + h - 2.9, title, size=7, color="white", weight="bold", z=3)
    tw, _ = extent(t)
    rbox(x + 1.5, y + h - 4.9, tw + 2.4, 4.0, fc=col, r=1.0, z=2)
    return x + 1.5 + tw + 2.4


def oplus(x, y, r=1.25, col=ARROW):
    ax.add_patch(Circle((x, y), r, fc="white", ec=col, lw=0.6, zorder=5))
    ax.plot([x - r * 0.6, x + r * 0.6], [y, y], color=col, lw=0.6, zorder=6, solid_capstyle="butt")
    ax.plot([x, x], [y - r * 0.6, y + r * 0.6], color=col, lw=0.6, zorder=6, solid_capstyle="butt")


def cuboid(x, y, w, h, d, base, text=None, z=4):
    """A layer drawn as a block: front face, top face and right face (depth d along a 37-degree axis)."""
    dx, dy = d * 0.8, d * 0.6
    edge = mix(base, "black", 0.25)
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, fc=mix(base, "white", 0.55), ec=edge, lw=0.4, zorder=z, joinstyle="round"))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, fc=mix(base, "black", 0.05), ec=edge, lw=0.4, zorder=z, joinstyle="round"))
    ax.add_patch(Rectangle((x, y), w, h, fc=mix(base, "white", 0.25), ec=edge, lw=0.4, zorder=z))
    if text:
        T(x + w / 2, y + h / 2, text, size=6, color="white", ha="center", weight="bold", z=z + 1)


def hatch_rect(x, y, w, h, fc, ec, step=0.55, lw=0.35, z=4):
    """A rectangle filled with diagonal lines drawn as segments, so that the hatching stays vector in PowerPoint."""
    rect(x, y, w, h, fc, ec=ec, lw=0.3, z=z)
    c = y - (x + w)                          # lines y' = x' + c, from the lower-right to the upper-left corner
    while c < y + h - x:
        xa, xb = max(x, y - c), min(x + w, y + h - c)
        if xb > xa:
            ax.plot([xa, xb], [xa + c, xb + c], color=ec, lw=lw, zorder=z + 0.1, solid_capstyle="butt")
        c += step


def vector(x, y, w, h, n, colors, ec, z=4):
    """A column of n cells (a layer's units or the ten task outputs)."""
    ch = h / n
    for k in range(n):
        rect(x, y + k * ch, w, ch, fc=colors[k % len(colors)], ec=ec, lw=0.3, z=z)


# ---------------------------------------------------------------- icons
def droplet(cx, cy, s, col):
    v = [(0, 0.5), (0.05, 0.3), (0.34, 0.05), (0.34, -0.17), (0.34, -0.4), (0.18, -0.5), (0, -0.5), (-0.18, -0.5),
         (-0.34, -0.4), (-0.34, -0.17), (-0.34, 0.05), (-0.05, 0.3), (0, 0.5), (0, 0.5)]
    codes = [Path.MOVETO] + [Path.CURVE4] * 12 + [Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path([(cx + a * s, cy + b * s) for a, b in v], codes), fc=col, ec="none", zorder=6))


def sun(cx, cy, s, col):
    ax.add_patch(Circle((cx, cy), 0.22 * s, fc=col, ec="none", zorder=6))
    for k in range(8):
        a = k * np.pi / 4
        ax.plot([cx + 0.32 * s * np.cos(a), cx + 0.5 * s * np.cos(a)], [cy + 0.32 * s * np.sin(a), cy + 0.5 * s * np.sin(a)],
                color=col, lw=0.7, zorder=6, solid_capstyle="round")


def snowflake(cx, cy, s, col):
    for k in range(3):
        a = k * np.pi / 3 + np.pi / 2
        ax.plot([cx - 0.5 * s * np.cos(a), cx + 0.5 * s * np.cos(a)], [cy - 0.5 * s * np.sin(a), cy + 0.5 * s * np.sin(a)],
                color=col, lw=0.7, zorder=6, solid_capstyle="round")
    for k in range(6):
        a = k * np.pi / 3 + np.pi / 2
        bx, by = cx + 0.3 * s * np.cos(a), cy + 0.3 * s * np.sin(a)
        for sgn in (-1, 1):
            b = a + sgn * np.pi / 4
            ax.plot([bx, bx + 0.15 * s * np.cos(b)], [by, by + 0.15 * s * np.sin(b)], color=col, lw=0.6, zorder=6,
                    solid_capstyle="round")


def crystals(cx, cy, s, col):
    q = 0.34 * s
    for ox, oy in ((-0.2, -0.2), (0.2, -0.2), (0.0, 0.17)):
        ax.add_patch(FancyBboxPatch((cx + ox * s - q / 2, cy + oy * s - q / 2), q, q, boxstyle=f"round,pad=0,rounding_size={q * 0.18}",
                                    fc=col, ec="white", lw=0.3, zorder=6))


def flask(cx, cy, s, col):
    """An Erlenmeyer flask with solution (alkaline stress)."""
    body = [(-0.11, 0.46), (-0.11, 0.12), (-0.42, -0.42), (-0.36, -0.5), (0.36, -0.5), (0.42, -0.42), (0.11, 0.12), (0.11, 0.46)]
    ax.add_patch(Polygon([(cx + a * s, cy + b * s) for a, b in body], closed=True, fc="white", ec=col, lw=0.6, zorder=6,
                         joinstyle="round"))
    liquid = [(-0.236, -0.1), (-0.42, -0.42), (-0.36, -0.5), (0.36, -0.5), (0.42, -0.42), (0.236, -0.1)]
    ax.add_patch(Polygon([(cx + a * s, cy + b * s) for a, b in liquid], closed=True, fc=col, ec=col, lw=0.6, zorder=7,
                         joinstyle="round"))
    ax.plot([cx - 0.19 * s, cx + 0.19 * s], [cy + 0.5 * s, cy + 0.5 * s], color=col, lw=0.8, zorder=7, solid_capstyle="round")


ICON = {"drought": droplet, "salt": crystals, "heat": sun, "cold": snowflake, "alkaline": flask}


def icon(stress, cx, cy, s):
    ICON[stress](cx, cy, s, fs.STRESS_COL[stress])


def padlock(cx, cy, s, col):
    rbox(cx - 0.36 * s, cy - 0.42 * s, 0.72 * s, 0.52 * s, fc=col, r=0.08 * s, z=7)
    ax.add_patch(Arc((cx, cy + 0.1 * s), 0.46 * s, 0.6 * s, theta1=0, theta2=180, lw=0.7, color=col, zorder=7))
    ax.plot([cx - 0.23 * s, cx - 0.23 * s], [cy + 0.1 * s, cy + 0.02 * s], color=col, lw=0.7, zorder=7)
    ax.plot([cx + 0.23 * s, cx + 0.23 * s], [cy + 0.1 * s, cy + 0.02 * s], color=col, lw=0.7, zorder=7)


def pod(cx, cy, s, col):
    ax.add_patch(Ellipse((cx, cy), 1.0 * s, 0.42 * s, angle=18, fc=col, ec=mix(col, "black", 0.2), lw=0.4, zorder=6))
    for k in (-1, 0, 1):
        ax.add_patch(Circle((cx + k * 0.27 * s * np.cos(np.radians(18)), cy + k * 0.27 * s * np.sin(np.radians(18))),
                            0.11 * s, fc=mix(col, "white", 0.55), ec="none", zorder=7))


LOGO_FONT = FontProperties(family="DejaVu Sans", weight="bold")


def letter(ch, x, y, w, h, col):
    tp = TextPath((0, 0), ch, size=1, prop=LOGO_FONT)
    bb = tp.get_extents()
    tr = Affine2D().translate(-bb.x0, -bb.y0).scale(w / bb.width, h / bb.height).translate(x, y)
    ax.add_patch(PathPatch(tr.transform_path(tp), fc=col, ec="none", zorder=6))


# ======================================================================================== top left: inputs per gene
region(2, 96, 83, 38, "Inputs per gene", "#00796B", "#eaf6f2")
yg = 124.0
ax.plot([5, 82], [yg, yg], color="#9aa3ad", lw=0.8, zorder=3, solid_capstyle="butt")
TSS_X, TTS_X = 26.0, 68.0
for x0, x1 in ((16.0, 31.0), (63.0, 78.0)):                       # the two window halves
    rbox(x0, yg - 2.9, x1 - x0, 5.8, fc="#cdebe3", ec=fs.TEAL, lw=0.6, r=0.8, z=2)
UTR, CDS = "#9aa5b1", "#3d4a5c"
rect(TSS_X, yg - 0.7, 2.5, 1.4, UTR, z=4)                         # 5' UTR
for x0, x1 in ((28.5, 33.0), (37.0, 42.0), (46.5, 51.5), (56.0, 64.5)):
    rect(x0, yg - 1.4, x1 - x0, 2.8, CDS, z=4)                    # coding exons
rect(64.5, yg - 0.7, TTS_X - 64.5, 1.4, UTR, z=4)                 # 3' UTR
ax.plot([TSS_X, TSS_X, TSS_X + 3.2], [yg + 1.4, yg + 4.4, yg + 4.4], color=fs.INK, lw=0.7, zorder=5, solid_joinstyle="miter")
arrow(TSS_X + 2.4, yg + 4.4, TSS_X + 4.2, yg + 4.4, col=fs.INK, lw=0.7, ms=4)
T(TSS_X + 4.8, yg + 4.4, "TSS", ha="left")
ax.plot([TTS_X, TTS_X], [yg - 1.8, yg + 3.4], color=fs.INK, lw=0.7, zorder=5)
T(TTS_X, yg + 4.6, "TTS", ha="center")
for x, s in ((16.0, "−1,000"), (31.0, "+500"), (63.0, "−500"), (78.0, "+1,000")):
    T(x, yg - 4.1, s, ha="center", va="top", color=fs.INK2)

# the window assembled for the networks: two 1,500-bp halves and a 20-bp spacer, coding sequence masked
ys, hs = 108.6, 3.6
S1, S2, SL = 8.0, 44.6, 35.0                                      # starts of the halves; 1,500 bp = 35 mm
for (wx0, wx1), sx in (((16.0, 31.0), S1), ((63.0, 78.0), S2)):
    ax.plot([wx0, sx], [yg - 6.2, ys + hs + 0.3], color="#9fb8b1", lw=0.5, ls=(0, (2, 1.5)), zorder=2)
    ax.plot([wx1, sx + SL], [yg - 6.2, ys + hs + 0.3], color="#9fb8b1", lw=0.5, ls=(0, (2, 1.5)), zorder=2)
NS = 100
bases = rng.choice(list("ACGT"), size=2 * NS)
for half, sx in enumerate((S1, S2)):
    for k in range(NS):
        rect(sx + k * SL / NS, ys, SL / NS + 0.02, hs, BASE[bases[half * NS + k]], z=3, lw=0)
mask = [(S1 + SL * (1250 / 1500), S1 + SL), (S2, S2 + SL * (150 / 1500))]   # coding sequence in each half
for x0, x1 in mask:
    hatch_rect(x0, ys, x1 - x0, hs, "#e3e3e3", "#8c8c8c")
rect(S1 + SL, ys, S2 - S1 - SL, hs, "white", ec="#8c8c8c", lw=0.3, z=4)
for x in (S1 + SL * (1000 / 1500), S2 + SL * (500 / 1500)):
    ax.plot([x, x], [ys - 0.7, ys + hs + 0.7], color=fs.INK, lw=0.7, zorder=5)
T(S1 + SL / 2, ys - 1.3, "TSS half, 1,500 bp", ha="center", va="top", color=fs.INK2)
T(S2 + SL / 2, ys - 1.3, "TTS half, 1,500 bp", ha="center", va="top", color=fs.INK2)
T((S1 + SL + S2) / 2, ys + hs + 1.5, "20 N", ha="center", va="bottom", color=fs.INK2)
T((S1 + SL + S2) / 2, ys - 1.3, "coding sequence\n(masked)", ha="center", va="top", color=fs.INK2, linespacing=1.05)

# gene features
yf = 97.9
T(5, yf + 3.6, "15 gene features", weight="bold", color="#00695C")
for k in range(15):
    rect(5 + k * 1.62, yf - 0.3, 1.5, 1.9, mix(GF_C, "white", 0.15 + 0.6 * ((k * 7) % 15) / 15), ec="white", lw=0.2, z=4)
T(31.0, yf + 2.9, "expression level, breadth and tau in the Plant Gene Atlas;", color=fs.INK2)
T(31.0, yf + 0.5, "gene structure, GC content, family size, duplication", color=fs.INK2)

# ======================================================================================== top right: labels
xr = region(88, 96, 84, 38, "Stress-response labels", "#34495E", "#eef2f7")
T(xr + 2.0, 131.1, "81 studies, 1,717 runs, 118 stress-versus-control comparisons", color=fs.INK2)
for k, s in enumerate(fs.STRESS):
    yy = 124.6 - 4.4 * k
    icon(s, 91.6, yy, 3.0)
    T(94.4, yy, fs.STRESS_LABEL[s])
    n = int(cnt.loc[s, "contrasts"])
    rbox(105.0, yy - 1.0, n * 0.2, 2.0, fc=fs.COMPARISON, r=0.3, z=4)
    T(105.0 + n * 0.2 + 0.6, yy, str(n), color=fs.INK2)
T(109.6, 103.8, "comparisons", ha="center", va="top", color=fs.INK2)
arrow(117.6, 115.8, 120.8, 115.8)
# volcano plot of one comparison (schematic)
vx0, vy0, vw, vh = 122.0, 107.0, 14.0, 19.5
ax.plot([vx0, vx0 + vw], [vy0, vy0], color=fs.INK2, lw=0.5, zorder=4)
ax.plot([vx0, vx0], [vy0, vy0 + vh], color=fs.INK2, lw=0.5, zorder=4)
lfc = rng.normal(0, 1.0, 140)
lp = np.abs(lfc) ** 1.6 * 3.2 + rng.exponential(1.2, 140)
for a, b in zip(lfc, lp):
    col = fs.UP if (a > 1.0 and b > 4) else (fs.DOWN if (a < -1.0 and b > 4) else "#b3b3b3")
    ax.add_patch(Circle((vx0 + vw / 2 + a * 2.4, vy0 + 0.8 + min(b, 17.5)), 0.32, fc=col, ec="none", zorder=5))
T(vx0 + vw / 2, 103.8, "DESeq2", ha="center", va="top", color=fs.INK2)
arrow(137.4, 115.8, 140.4, 115.8)
# forest plot: studies and the random-effects estimate (schematic)
fx0 = 145.0
ax.plot([fx0, fx0], [109.5, 126.5], color=fs.INK2, lw=0.5, ls=(0, (2, 1.2)), zorder=4)
for k, (m, w_) in enumerate(((3.6, 2.2), (2.2, 1.6), (4.3, 2.8), (2.9, 1.3))):
    yy = 125.2 - 3.3 * k
    ax.plot([fx0 + m - w_, fx0 + m + w_], [yy, yy], color=fs.INK2, lw=0.6, zorder=5)
    rect(fx0 + m - 0.55, yy - 0.55, 1.1, 1.1, fs.UP, z=6)
ax.add_patch(Polygon([(fx0 + 1.9, 110.8), (fx0 + 3.2, 111.8), (fx0 + 4.5, 110.8), (fx0 + 3.2, 109.8)], closed=True,
                     fc=fs.UP, ec="none", zorder=6))
T(fx0 + 1.8, 103.8, "vote + meta-analysis", ha="center", va="top", color=fs.INK2)
arrow(152.0, 115.8, 155.0, 115.8)
# gene x stress labels
CALL = {"u": fs.UP, "d": fs.DOWN, "n": "#cfcfcf", "-": "white"}
calls = ["unnnn", "nunun", "dnnd-", "nn-nn", "uuu-n", "nndnn", "d-nnd", "nnnnn", "-nunn"]
mx0, mcw, mch = 156.6, 2.65, 1.9
for r_, row in enumerate(calls):
    for c_, cl in enumerate(row):
        rect(mx0 + c_ * mcw, 107.0 + r_ * mch, mcw, mch, CALL[cl], ec="white" if cl != "-" else "#c8c8c8", lw=0.3, z=4)
for c_, s in enumerate(fs.STRESS):
    icon(s, mx0 + (c_ + 0.5) * mcw, 126.2, 2.2)
T(mx0 + 2.5 * mcw, 103.8, "gene labels", ha="center", va="top", color=fs.INK2)
lx = 91.0
for col, lab in ((fs.UP, "up-regulated"), (fs.DOWN, "down-regulated"), ("#cfcfcf", "non-responsive"), ("white", "no label")):
    rect(lx, 97.3, 2.0, 2.0, col, ec="#9a9a9a" if col == "white" else "none", lw=0.3, z=4)
    t = T(lx + 2.7, 98.3, lab, color=fs.INK2)
    lx += 2.7 + extent(t)[0] + 3.4

# ======================================================================================== middle: StressCRE
region(2, 46, 170, 46, "StressCRE", "#C63D2B", "#fdf1ef")
flow(11.6, 95.8, 11.6, 92.3)
XO = 84.0                                   # where the gene-feature embedding joins both networks
YC, YP = 82.2, 56.3                         # rows of the convolutional and PlantCAD2 networks
# inputs
oh_x, oh_y, cw = 5.0, 80.0, 1.1
_ = rng.choice(4, size=12)                  # (draw kept so the other schematics stay unchanged)
oh = ["ACGT".index(c) for c in "CATACGTGTCAG"]   # the same window as the PlantCAD2 input
for k, b in enumerate(oh):
    for r_ in range(4):
        rect(oh_x + k * cw, oh_y + (3 - r_) * cw, cw, cw, BASE["ACGT"[r_]] if r_ == b else "white", ec="#d0d0d0", lw=0.2, z=4)
T(oh_x + 6 * cw, oh_y - 0.8, "one-hot window\n(3,020 × 4)", ha="center", va="top", color=fs.INK2, linespacing=1.05)
seq = "TACGTGTC"                             # an ABRE (ACGTGT) in the context of the top learned motif, no G-box
# letters set at their Times New Roman Bold advance widths (em): the PowerPoint import joins them into one text box at
# these widths, so the figure and its editable copy agree
ADV = {"A": 0.72217, "C": 0.72217, "G": 0.77783, "T": 0.66699}
xs_ = 4.8
for ch in seq:
    T(xs_, YP, ch, size=6.5, color=BASE[ch], ha="left", weight="bold")
    xs_ += ADV[ch] * 6.5 / 72 * 25.4
T(11.6, 58.6, "window sequence", ha="center", va="bottom", color=fs.INK2)
for k in range(16):
    rect(5 + k * 0.8, 68.0, 0.8, 1.6, mix(GF_C, "white", 0.15 + 0.6 * ((k * 7) % 16) / 16), ec="white", lw=0.15, z=4)
T(11.4, 67.2, "gene features", ha="center", va="top", color=fs.INK2)

# convolutional network
T(24.0, 89.6, "Convolutional network, trained from scratch", size=6.5, color=CNN_D, weight="bold")
arrow(18.6, YC, 23.2, YC)
blocks = [(24.0, YC - 3.2, 10.0, 6.4, 2.4, "64", "377 × 64"), (39.4, YC - 4.0, 7.6, 8.0, 3.4, "128", "47 × 128"),
          (52.6, YC - 3.2, 5.0, 6.4, 2.4, "64", "5 × 64")]
for k, (x, y, w, h, d, ch, shape) in enumerate(blocks):
    cuboid(x, y, w, h, d, CNN_C, text=ch)
    T(x + w / 2, YC - 5.3, shape, ha="center", va="top", color=MUTE)
    if k < 2:
        arrow(x + w + d * 0.8 + 0.6, YC, blocks[k + 1][0] - 0.6, YC)
T(40.0, YC - 8.0, "per block: 2 convolutions (kernel 8), max-pooling", ha="center", va="top", color=fs.INK2)
arrow(60.8, YC, XO - 1.3, YC)
T((60.8 + XO) / 2, YC + 1.0, "flatten (320)", ha="center", va="bottom", color=MUTE)
oplus(XO, YC)
arrow(XO + 1.3, YC, XO + 2.9, YC)
vector(XO + 3.0, YC - 4.7, 1.8, 9.4, 8, [mix(CNN_C, "white", f) for f in (0.35, 0.6, 0.45, 0.7)], mix(CNN_C, "black", 0.2))
arrow(XO + 4.9, YC, XO + 7.1, YC)
vector(XO + 7.2, YC - 3.3, 1.8, 6.6, 6, [mix(CNN_C, "white", f) for f in (0.5, 0.3, 0.65)], mix(CNN_C, "black", 0.2))
T(XO + 3.9, YC - 5.3, "128", ha="center", va="top", color=MUTE)
T(XO + 8.1, YC - 5.3, "64", ha="center", va="top", color=MUTE)

# gene-feature networks: each of the two networks has its own (16 inputs: the 15 gene features and an indicator for genes
# without tau; 16 -> 64 -> 32 units)
YM = 68.8
GFD = mix(GF_C, "black", 0.1)


def mlp(yc):
    pts = []
    for x, n in zip((66.4, 71.0, 75.6), (3, 4, 2)):
        pts.append([(x, yc + (j - (n - 1) / 2) * 1.3) for j in range(n)])
    for a_, b_ in zip(pts[:-1], pts[1:]):
        for p in a_:
            for q in b_:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=mix(GF_C, "white", 0.5), lw=0.3, zorder=4)
    for layer in pts:
        for p in layer:
            ax.add_patch(Circle(p, 0.48, fc=GF_C, ec="white", lw=0.3, zorder=5))


YMC, YMP = 72.0, 65.6
mlp(YMC)
mlp(YMP)
ax.plot([18.6, 62.6], [YM, YM], color=GFD, lw=0.7, zorder=4)
ax.plot([62.6, 62.6], [YMP, YMC], color=GFD, lw=0.7, zorder=4)
arrow(62.6, YMC, 65.6, YMC, col=GFD)
arrow(62.6, YMP, 65.6, YMP, col=GFD)
T(61.6, YM - 0.9, "16 → 64 → 32, one per network", ha="right", va="top", color=MUTE)
for yy, yt in ((YMC, YC - 1.3), (YMP, YP + 1.3)):
    ax.plot([76.1, XO], [yy, yy], color=GFD, lw=0.7, zorder=4)
    arrow(XO, yy, XO, yt, col=GFD)

# PlantCAD2 network
T(24.0, 63.0, "PlantCAD2 network", size=6.5, color=PC2_D, weight="bold")
arrow(19.0, YP, 23.0, YP)
for k in range(4, -1, -1):
    rbox(23.2 + k * 0.7, 51.6 + k * 0.55, 18.0, 7.0, fc=mix(PC2_C, "white", 0.78 - 0.08 * k), ec=mix(PC2_C, "white", 0.2),
         lw=0.4, r=0.8, z=4 + (4 - k) * 0.01)
T(32.2, 56.4, "PlantCAD2-Small", color=PC2_D, ha="center", weight="bold", z=7)
T(32.2, 53.9, "24 layers", ha="center", color=PC2_D, z=7)
padlock(25.0, 49.4, 2.2, fs.INK2)
T(26.7, 49.3, "pretrained, frozen", ha="left", color=fs.INK2)
YB = 52.4                                   # baseline of the head elements
arrow(44.4, 55.2, 46.1, 55.2)
bx0, bcw, bch = 46.4, 0.92, 1.4
vals = rng.random((4, 12))
for r_ in range(4):
    for c_ in range(12):
        rect(bx0 + c_ * bcw, YB + r_ * bch, bcw, bch, mix(PC2_C, "white", 0.15 + 0.75 * vals[r_, c_]), ec="white", lw=0.15, z=4)
bxc = bx0 + 6 * bcw
T(bxc, 59.0, "60 × 1,536", ha="center", va="bottom", color=MUTE)
T(bxc, 51.4, "mean over\n50-bp bins", ha="center", va="top", color=fs.INK2, linespacing=1.05)
arrow(bx0 + 12 * bcw + 0.4, 55.2, 60.2, 55.2)
rbox(60.4, YB, 8.4, 5.6, fc=mix(PC2_C, "white", 0.25), ec=mix(PC2_C, "black", 0.2), lw=0.4, r=0.6, z=4)
T(64.6, 55.2, "conv ×2", ha="center", color="white", weight="bold")
T(64.6, 59.0, "60 × 128", ha="center", va="bottom", color=MUTE)
T(64.6, 51.4, "projection,\nresidual conv", ha="center", va="top", color=fs.INK2, linespacing=1.05)
arrow(69.2, 55.2, 71.5, 55.2)
att = [0.15, 0.25, 0.9, 0.35, 0.2, 0.15, 0.55, 0.3, 0.12, 0.2]
for k, a_ in enumerate(att):
    rect(71.9 + k * 0.85, YB, 0.7, 0.4 + a_ * 5.0, mix(PC2_C, "white", 0.6 - 0.5 * a_), z=4)
ax.plot([71.7, 80.5], [YB, YB], color=fs.INK2, lw=0.4, zorder=5)
T(76.1, 59.0, "10 × 128", ha="center", va="bottom", color=MUTE)
T(76.3, 51.4, "attention\nper task", ha="center", va="top", color=fs.INK2, linespacing=1.05)
arrow(80.8, YP, XO - 1.3, YP)
oplus(XO, YP)
arrow(XO + 1.3, YP, XO + 2.9, YP)
vector(XO + 3.0, YP - 3.3, 1.8, 6.6, 6, [mix(PC2_C, "white", f) for f in (0.5, 0.3, 0.65)], mix(PC2_C, "black", 0.2))
T(XO + 3.9, YP + 3.8, "10 × 64", ha="center", va="bottom", color=MUTE)

# outputs, mean and prediction
XV = 99.0                                   # the two output vectors (ten tasks)
p_cnn = [0.82, 0.08, 0.64, 0.05, 0.31, 0.10, 0.47, 0.06, 0.22, 0.04]
p_pc2 = [0.74, 0.11, 0.58, 0.07, 0.39, 0.06, 0.52, 0.09, 0.18, 0.05]
for (y0, p, col) in ((YC - 4.8, p_cnn, CNN_C), (YP - 4.8, p_pc2, PC2_C)):
    vector(XV, y0, 2.0, 9.6, 10, [mix(col, "white", 0.92 - 0.85 * v) for v in p][::-1], mix(col, "black", 0.2))
T(XV + 1.0, YC - 5.3, "10 tasks", ha="center", va="top", color=MUTE)
T(XV + 1.0, YP + 5.2, "10 tasks", ha="center", va="bottom", color=MUTE)
arrow(XO + 9.2, YC, XV - 0.2, YC)
arrow(XO + 4.9, YP, XV - 0.2, YP)
XM = 110.5
ax.add_patch(Circle((XM, YM), 3.4, fc="white", ec=fs.RED, lw=0.9, zorder=5))
T(XM, YM, "mean", ha="center", color=CNN_D, weight="bold", z=7)
arrow(XV + 2.3, YC, XM - 2.1, YM + 2.6, cs="arc3,rad=-0.15")
arrow(XV + 2.3, YP, XM - 2.1, YM - 2.6, cs="arc3,rad=0.15")
arrow(XM + 3.6, YM, 133.2, YM, col=fs.RED, lw=0.9)
T((XM + 133.2) / 2 + 1.8, YM + 1.0, "predicted\nprobabilities", ha="center", va="bottom", color=fs.INK2, linespacing=1.05)
# the ten tasks of one gene
gx0, gcw, gch = 140.0, 5.6, 4.6
pm = (np.array(p_cnn) + np.array(p_pc2)) / 2
for c_, s in enumerate(fs.STRESS):
    icon(s, gx0 + (c_ + 0.5) * gcw, 77.4, 3.0)
    for r_, d in enumerate(("up", "down")):
        v = pm[2 * c_ + r_]
        rect(gx0 + c_ * gcw, 69.6 - r_ * gch, gcw, gch, mix(fs.RED, "white", 0.92 - 0.85 * v), ec="white", lw=0.6, z=4)
T(gx0 - 0.8, 69.6 + gch / 2, "up", ha="right", color=fs.INK)
T(gx0 - 0.8, 69.6 - gch / 2, "down", ha="right", color=fs.INK)
gc = gx0 + 2.5 * gcw
T(gc, 63.4, "StressCRE prediction for one gene", ha="center", va="top", color=fs.INK, weight="bold")
T(gc, 60.6, "ten tasks: up- or down-regulated", ha="center", va="top", color=fs.INK2)
T(gc, 58.0, "versus non-responsive genes", ha="center", va="top", color=fs.INK2)
flow(gc, 95.8, gc, 81.4)
T(gc - 1.6, 87.6, "training labels", ha="right", color=fs.INK2)

# ======================================================================================== bottom: evaluation and analyses
cards = [(2.0, "Test design and accuracy", "#9A7500", "#fbf6e4"), (45.4, "Learned elements", fs.UP, "#fcefe5"),
         (88.8, "WGD paralogs", "#5E50B0", "#f1eff9"), (132.2, "Natural variation", "#3C8D2F", "#ecf5e9")]
CW = 39.8
for x, title, col, tint in cards:
    region(x, -5.0, CW, 47.0, title, col, tint)
    flow(x + CW / 2, 45.8, x + CW / 2, 42.3)

# test design: folds that keep related genes together, and accuracy
x0 = 4.4
for k in range(5):
    test = k == 4
    rbox(x0 + k * 7.0, 28.4, 6.0, 7.0, fc="white" if test else "#ece3c4", ec=fs.RED if test else "#c9b77a",
         lw=0.8 if test else 0.4, r=0.8, z=3)
    for j in range(3):
        cy = 33.95 - j * 2.2
        n = 2 + (k + j) % 2
        rbox(x0 + k * 7.0 + 0.8, cy - 0.8, 4.4, 1.6, fc="#f7f1dc" if not test else "#fdeceb", ec="#b9a463" if not test else fs.RED,
             lw=0.3, r=0.6, z=4)
        for i in range(n):
            ax.add_patch(Circle((x0 + k * 7.0 + 1.9 + i * 1.15, cy), 0.42, fc="#7d6a2e" if not test else fs.RED, ec="none", zorder=5))
T(x0 + 1.5 * 7.0 + 3.0, 27.7, "training folds", ha="center", va="top", color="#7d6a2e")
T(x0 + 4 * 7.0 + 3.0, 27.7, "test fold", ha="center", va="top", color=CNN_D, weight="bold")
yb = para(3.8, 25.0, "Protein families, WGD pairs and similar windows kept in one fold; 5 folds × 3 assignments.",
          width=37.0, step=2.7)
T(3.8, yb - 0.8, "Expression-matched AUPRC", va="top", color=fs.INK, weight="bold")
YBAR = yb - 6.0
for k, m in enumerate(("cand3", "stack", "cov", "kmer")):
    yy = YBAR - 3.8 * k
    lab, col, _ = fs.MODEL[m]
    T(3.8, yy, lab, color=fs.INK)
    rbox(25.0, yy - 1.0, ACC[m] * 62, 2.0, fc=col, r=0.3, z=4)
    T(25.0 + ACC[m] * 62 + 0.6, yy, f"{ACC[m]:.3f}", color=fs.INK, weight="bold" if m == "cand3" else "normal")

# learned elements: contribution scores of an ABRE and its context
# illustrative heights, shaped like the mean contribution of the convolutional network at ABRE sites of drought-induced genes
# (relative to the highest base: A 0.56, C 0.98, G 1.00, T 0.60, G 0.61, G/T 0.21; flanks near zero)
logo = "GCATACGTGTCAG"
hts = [0.6, 0.7, 0.8, 0.8, 4.1, 7.0, 7.2, 4.3, 4.4, 1.9, 0.8, 0.6, 0.7]
lx0, lw_ = 48.4, 2.55
for k, (ch, h) in enumerate(zip(logo, hts)):
    letter(ch, lx0 + k * lw_ + 0.12, 27.2, lw_ - 0.24, h, BASE[ch] if 4 <= k <= 9 else "#b9b9b9")   # the six ABRE bases in colour
ax.plot([lx0, lx0 + len(logo) * lw_], [27.0, 27.0], color=fs.INK2, lw=0.5, zorder=5)
bx_0, bx_1 = lx0 + 4 * lw_, lx0 + 10 * lw_
ax.plot([bx_0, bx_0, bx_1, bx_1], [25.9, 25.2, 25.2, 25.9], color=fs.UP, lw=0.7, zorder=5)
T((bx_0 + bx_1) / 2, 24.6, "ABRE (ACGTGG/T)", ha="center", va="top", color=fs.UP, weight="bold")
T(lx0 + len(logo) * lw_ / 2 + 0.4, 36.8, "contribution scores", ha="center", va="bottom", color=fs.INK2) if False else None
yb = para(47.2, 19.0, "The ABRE, the main learned element of induction by all five stresses, carried 3–11% of the positive contribution in "
                     "0.3–0.4% of the bases.", step=3.0)
para(47.2, yb - 0.6, "ABREs inserted in silico had their largest predicted effect near the TSS; two copies acted more than "
                     "additively.", step=3.0)

# WGD paralogs: two copies, one responsive and tissue-specific
for k, (yy, resp) in enumerate(((32.6, True), (26.4, False))):
    T(90.4, yy, f"copy {k + 1}", color=fs.INK2)
    ax.plot([97.6, 106.6], [yy, yy], color="#9aa3ad", lw=0.8, zorder=3)
    rect(100.2, yy - 1.3, 5.6, 2.6, CDS, z=4)
    if resp:
        arrow(108.4, yy - 1.6, 108.4, yy + 2.4, col=fs.UP, lw=1.1, ms=6)
    else:
        ax.plot([107.6, 109.4], [yy, yy], color="#8c8c8c", lw=1.6, zorder=5, solid_capstyle="round")
    expr = [0.3, 0.2, 4.0, 0.4, 0.2, 0.3] if resp else [2.0, 2.3, 1.8, 2.2, 2.4, 1.9]
    for j, e in enumerate(expr):
        rect(113.6 + j * 2.05, yy - 1.6, 1.6, e, GF_C, z=4)
    ax.plot([113.2, 126.2], [yy - 1.6, yy - 1.6], color=fs.INK2, lw=0.4, zorder=5)
ax.plot([103.0, 103.0], [27.9, 31.1], color=MUTE, lw=0.6, ls=(0, (1.5, 1)), zorder=5)
T(103.8, 29.5, "WGD", color=MUTE)
T(119.7, 24.0, "tissue expression", ha="center", va="top", color=MUTE)
yb = para(90.6, 19.0, "15,487 pairs; 335 divergent pair–task combinations: the responsive copy was the more "
                     "tissue-specific one in 61%.", step=3.0)
para(90.6, yb - 0.6, "With the same gene features for both copies, StressCRE did not predict which copy responded "
                     "(paired AUROC 0.54).", step=3.0)

# natural variation: SNPs in a window, one destroying an ABRE; wild soybean W05
yw = 29.6
pod(137.0, yw + 1.6, 4.0, "#A88B4A")          # a soybean pod in a neutral tan (accessions are wild and cultivated)
ax.plot([140.6, 169.6], [yw, yw], color="#9aa3ad", lw=1.6, zorder=3, solid_capstyle="butt")
rect(150.0, yw - 1.0, 4.6, 2.0, fs.UP, z=4)
for xs_, hh, hit in ((143.0, 2.6, False), (146.6, 3.4, False), (152.3, 4.2, True), (158.4, 3.0, False), (163.0, 3.6, False),
                     (166.8, 2.4, False)):
    ax.plot([xs_, xs_], [yw + 1.0, yw + hh], color=fs.DOWN if hit else fs.INK2, lw=0.5, zorder=4)
    ax.add_patch(Circle((xs_, yw + hh + 0.5), 0.6, fc=fs.DOWN if hit else "white", ec=fs.DOWN if hit else fs.INK2, lw=0.5,
                        zorder=5))
T(152.3, yw - 1.6, "ABRE lost: ACGTGG → ACATGG", ha="center", va="top", color=fs.DOWN, weight="bold")
yb = para(134.0, 24.4, "1,511 accessions. Among 300,000 scored SNP–gene combinations, large-effect SNPs (top 1%) "
                      "were enriched for ABRE gains and losses but not more often rare. A test without predicted effects, over "
                      "3.2 million combinations, found that SNPs destroying an ABRE were more often rare.", step=2.7)
para(134.0, yb - 0.6, "Wild soybean W05, salt up-regulation: AUROC 0.80 (tau alone 0.77); its own windows changed "
                      "the predictions little.", step=2.7)

fs.save(fig, f"{P}/figures", "fig1_overview")
print("written figures/fig1_overview")
