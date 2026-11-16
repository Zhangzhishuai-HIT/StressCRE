"""W05 windows anchored at the TSS and TTS homologous to the Wm82.a4 ones (cross-genome test, plan step 8).

The W05 annotation mostly lacks UTRs, so windows cut at its annotated gene ends are shifted against the Wm82
windows. Instead, for each one-to-one ortholog pair (work/w05/orthologs.tsv) the unmasked Wm82.a4 TSS half
(TSS-1000..TSS+499) and TTS half (TTS-499..TTS+1000), in transcript orientation, are aligned (edlib, infix) to the
W05 sequence within 3 kb of the W05 gene's annotated start and end; the W05 bases aligned to the Wm82 TSS and TTS
become the W05 anchors, and the W05 window is cut around them with the same layout (20 N spacer) and with every W05
CDS masked to N. Pairs whose half alignments have > 20% edits are dropped.

Usage: python -I w05_homwindows.py <project> <w05_genome.fa.gz> <w05_gff3.gz> <a4_genome.fa.gz>
Writes seqs/w05_homwindows.fa.gz (headers = Wm82.a4 gene ids) and seqs/w05_homwindows.tsv
(gene_a4, gene_w05, chrom, strand, tss, tts, edit_tss_half, edit_tts_half, window_edit_frac vs the Wm82 window).
"""
import gzip, re, sys
import edlib
import numpy as np
import pandas as pd

P, WFA, WGFF, AFA = sys.argv[1:5]
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
rc = lambda s: s.translate(COMP)[::-1]


def genome(fa):
    g, name, buf = {}, None, []
    for line in gzip.open(fa, "rt"):
        if line.startswith(">"):
            if name:
                g[name] = "".join(buf).upper()
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    g[name] = "".join(buf).upper()
    return g


gw, ga = genome(WFA), genome(AFA)
cds = {}
for line in gzip.open(WGFF, "rt"):
    f = line.split("\t")
    if len(f) > 8 and f[2] == "CDS":
        cds.setdefault(f[0], []).append((int(f[3]) - 1, int(f[4])))
masked = {}
for c, s in gw.items():
    a = np.frombuffer(s.encode(), np.uint8).copy()
    for s0, e0 in cds.get(c, []):
        a[s0:e0] = ord("N")
    masked[c] = a.tobytes().decode()


def grab(seqs, c, s1, e1):
    """1-based inclusive interval, N beyond the ends"""
    s, L = seqs[c], len(seqs[c])
    lo, hi = max(s1, 1), min(e1, L)
    return "N" * (lo - s1) + (s[lo - 1:hi] if lo <= hi else "") + "N" * (e1 - hi)


def oriented(seqs, c, strand, a1, a2):
    x = grab(seqs, c, a1, a2)
    return x if strand == "+" else rc(x)


win4 = pd.read_csv(f"{P}/seqs/wm82a4_windows.tsv", sep="\t").set_index("gene")
winw = pd.read_csv(f"{P}/seqs/w05_windows.tsv", sep="\t").set_index("gene")
w4seq = {}
for line in gzip.open(f"{P}/seqs/wm82a4_windows.fa.gz", "rt"):
    if line.startswith(">"):
        k = line[1:].strip()
    else:
        w4seq[k] = line.strip()
orth = pd.read_csv(f"{P}/work/w05/orthologs.tsv", sep="\t")
FL = 3000
out, rows, fails = [], [], 0


def anchor(query, q_idx, c, strand, center):
    """align query (transcript orientation) to W05 around `center` (1-based, transcript-oriented window
    [center-FL-1500, center+FL+1500]); return the 1-based genomic W05 position aligned to query[q_idx] and edit fraction"""
    if strand == "+":
        r1, r2 = center - FL - 1500, center + FL + 1500
    else:
        r1, r2 = center - FL - 1500, center + FL + 1500
    region = oriented(gw, c, strand, r1, r2)
    al = edlib.align(query, region, mode="HW", task="path")
    if al["editDistance"] < 0 or al["editDistance"] > 0.2 * len(query):
        return None, None
    t0 = al["locations"][0][0]
    qi, ti = 0, t0
    for n, op in re.findall(r"(\d+)([=XID])", al["cigar"]):
        n = int(n)
        if op in "=X":
            if qi + n > q_idx:
                ti += q_idx - qi
                qi = q_idx
                break
            qi += n; ti += n
        elif op == "I":             # bases in the query only
            if qi + n > q_idx:
                qi = q_idx
                break
            qi += n
        else:                       # D: bases in the target only
            ti += n
    pos = r1 + ti if strand == "+" else r2 - ti      # 1-based genomic coordinate
    return pos, al["editDistance"] / len(query)


for r in orth.itertuples():
    a, w = win4.loc[r.gene_a4], winw.loc[r.gene_w05]
    if a.strand == "+":
        q_tss = grab(ga, a.chrom, a.tss - 1000, a.tss + 499)
        q_tts = grab(ga, a.chrom, a.tts - 499, a.tts + 1000)
    else:
        q_tss = rc(grab(ga, a.chrom, a.tss - 499, a.tss + 1000))
        q_tts = rc(grab(ga, a.chrom, a.tts - 1000, a.tts + 499))
    tss_w, e1 = anchor(q_tss, 1000, w.chrom, w.strand, w.tss)
    tts_w, e2 = anchor(q_tts, 499, w.chrom, w.strand, w.tts)
    if tss_w is None or tts_w is None:
        fails += 1
        continue
    if w.strand == "+":
        s = grab(masked, w.chrom, tss_w - 1000, tss_w + 499) + "N" * 20 + grab(masked, w.chrom, tts_w - 499, tts_w + 1000)
    else:
        s = rc(grab(masked, w.chrom, tss_w - 499, tss_w + 1000)) + "N" * 20 + rc(grab(masked, w.chrom, tts_w - 1000, tts_w + 499))
    assert len(s) == 3020
    d = edlib.align(s, w4seq[r.gene_a4], mode="NW")["editDistance"] / 3020
    out.append(f">{r.gene_a4}\n{s}\n")
    rows.append((r.gene_a4, r.gene_w05, w.chrom, w.strand, tss_w, tts_w, round(e1, 4), round(e2, 4), round(d, 4)))
with gzip.open(f"{P}/seqs/w05_homwindows.fa.gz", "wt") as f:
    f.write("".join(out))
t = pd.DataFrame(rows, columns=["gene_a4", "gene_w05", "chrom", "strand", "tss", "tts", "edit_tss_half", "edit_tts_half", "window_edit_frac"])
t.to_csv(f"{P}/seqs/w05_homwindows.tsv", sep="\t", index=False)
print(f"{len(t)} homologous W05 windows; {fails} pairs dropped; window edit fraction quantiles",
      t.window_edit_frac.quantile([0.25, 0.5, 0.75, 0.9]).round(4).to_dict(), "; identical:", int((t.window_edit_frac == 0).sum()))
