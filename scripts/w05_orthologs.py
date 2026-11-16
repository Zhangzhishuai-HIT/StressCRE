"""One-to-one W05 (G. soja) - Wm82.a4 orthologs from the DIAMOND searches of jobs/w05_orthologs.sbatch, and how much
their model windows differ.

Ortholog = reciprocal best protein hit (primary isoforms), >= 90% identity over >= 80% of both proteins, on the
same chromosome number (W05 and Wm82 are collinear). Window difference = edit distance between the W05 and the
Wm82.a4 window (A/C/G/T/N strings of 3,020 bp; edlib, global alignment) divided by 3,020.

Usage: python -I w05_orthologs.py <project>    -> work/w05/orthologs.tsv (gene_a4, gene_w05, pident, window_edit_frac)
"""
import gzip, re, sys
import edlib
import pandas as pd

P = sys.argv[1]
W = f"{P}/work/w05"
cols = ["q", "s", "pident", "length", "qlen", "slen", "evalue", "bitscore"]
g = lambda x: re.sub(r"\.\d+$", "", x.split("ann1.")[-1])
a = pd.read_csv(f"{W}/w05_vs_a4.m8", sep="\t", names=cols)
b = pd.read_csv(f"{W}/a4_vs_w05.m8", sep="\t", names=cols)
a = a.sort_values("bitscore", ascending=False).drop_duplicates("q")
b = b.sort_values("bitscore", ascending=False).drop_duplicates("q")
a["gw"], a["g4"] = a.q.map(g), a.s.map(g)
b["g4"], b["gw"] = b.q.map(g), b.s.map(g)
m = a.merge(b[["g4", "gw"]], on=["g4", "gw"])
m = m[(m.pident >= 90) & (m.length >= 0.8 * m.qlen) & (m.length >= 0.8 * m.slen)]
m = m[m.gw.str.slice(8, 10) == m.g4.str.slice(6, 8)]          # Glysoja.01G... vs Glyma.01G...
m = m.drop_duplicates("g4").drop_duplicates("gw")


def read(fa):
    d = {}
    for line in gzip.open(fa, "rt"):
        if line.startswith(">"):
            k = line[1:].strip()
        else:
            d[k] = line.strip()
    return d


w4, ww = read(f"{P}/seqs/wm82a4_windows.fa.gz"), read(f"{P}/seqs/w05_windows.fa.gz")
m = m[m.g4.isin(w4) & m.gw.isin(ww)]
m["window_edit_frac"] = [edlib.align(ww[x], w4[y], mode="NW")["editDistance"] / 3020 for x, y in zip(m.gw, m.g4)]
out = m.rename(columns={"g4": "gene_a4", "gw": "gene_w05"})[["gene_a4", "gene_w05", "pident", "window_edit_frac"]]
out.to_csv(f"{W}/orthologs.tsv", sep="\t", index=False)
print(len(out), "ortholog pairs; window edit fraction quantiles:", out.window_edit_frac.quantile([0.25, 0.5, 0.75, 0.9]).round(4).to_dict(),
      "; identical windows:", int((out.window_edit_frac == 0).sum()))
