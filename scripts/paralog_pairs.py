"""Whole-genome-duplication paralog pairs in Wm82.a4 and how often their stress labels differ (pilot).

Pairs = reciprocal best protein hits (MMseqs2, >= 50% identity, >= 50% coverage) between genes on
different chromosomes that sit in collinear blocks: of the 10 genes on each side of gene A, at least
3 have one of their top-5 hits within 20 genes of gene B (a simplified MCScanX anchor rule).
Same-chromosome pairs within 10 genes are reported separately as tandem duplicates.

Usage: python3 -I paralog_pairs.py <self_search.m8> <a4.gff3.gz> <pilot_labels.tsv.gz> <out_dir>
"""
import collections, gzip, os, sys
import pandas as pd

M8, GFF, LAB, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
pos = {}
for line in gzip.open(GFF, "rt"):
    if line.startswith("#"):
        continue
    f = line.rstrip("\n").split("\t")
    if len(f) < 9 or f[2] != "gene":
        continue
    gid = [x for x in f[8].split(";") if x.startswith("Name=")][0][5:]
    pos[gid] = (f[0], int(f[3]))
order = {}
by_chr = collections.defaultdict(list)
for g, (c, s) in pos.items():
    by_chr[c].append((s, g))
for c in by_chr:
    by_chr[c].sort()
    for i, (_, g) in enumerate(by_chr[c]):
        order[g] = (c, i)
chr_genes = {c: [g for _, g in v] for c, v in by_chr.items()}

hits = collections.defaultdict(list)
for line in open(M8):
    q, t, pid, alnlen, ev, bits, qcov, tcov = line.split("\t")
    q = q.split("ann1.")[-1].rsplit(".", 1)[0]; t = t.split("ann1.")[-1].rsplit(".", 1)[0]
    if q == t:
        continue
    hits[q].append((float(bits), t, float(pid)))
for q in hits:
    hits[q].sort(reverse=True)
best = {q: h[0][1] for q, h in hits.items() if h}
top5 = {q: [t for _, t, _ in h[:5]] for q, h in hits.items()}
pid = {(q, t): p for q, h in hits.items() for _, t, p in h}

wgd, tandem = [], []
seen = set()
for a, b in best.items():
    if best.get(b) != a or (b, a) in seen or a not in order or b not in order:
        continue
    seen.add((a, b))
    ca, ia = order[a]; cb, ib = order[b]
    if ca == cb and abs(ia - ib) <= 10:
        tandem.append((a, b, pid.get((a, b), 0))); continue
    neigh = chr_genes[ca][max(0, ia - 10): ia] + chr_genes[ca][ia + 1: ia + 11]
    anchors = 0
    for n in neigh:
        for t in top5.get(n, []):
            if t in order and order[t][0] == cb and abs(order[t][1] - ib) <= 20:
                anchors += 1; break
    if anchors >= 3:
        wgd.append((a, b, pid.get((a, b), 0), anchors, ca == cb))
wd = pd.DataFrame(wgd, columns=["gene_a", "gene_b", "protein_identity", "anchors", "same_chrom"])
wd.to_csv(f"{OUT}/wgd_pairs.tsv", sep="\t", index=False)
pd.DataFrame(tandem, columns=["gene_a", "gene_b", "protein_identity"]).to_csv(f"{OUT}/tandem_pairs.tsv", sep="\t", index=False)
in_pairs = set(wd.gene_a) | set(wd.gene_b)
print(f"genes with a reciprocal best hit: {sum(1 for a, b in best.items() if best.get(b) == a)}")
print(f"WGD (collinear) pairs: {len(wd)}  genes in them: {len(in_pairs)} ({100 * len(in_pairs) / len(pos):.1f}% of {len(pos)})")
print(f"  protein identity median {wd.protein_identity.median():.1f}%, same-chromosome pairs {int(wd.same_chrom.sum())}")
print(f"tandem pairs: {len(tandem)}")

lab = pd.read_csv(LAB, sep="\t")
lab["call"] = "unclear"
lab.loc[(lab.vote == "up") & (lab.meta == "up"), "call"] = "up"
lab.loc[(lab.vote == "down") & (lab.meta == "down"), "call"] = "down"
lab.loc[(lab.meta == "nonresp") & ~lab.vote.isin(["up", "down"]), "call"] = "nonresp"
rows = []
for s, l in lab.groupby("stress"):
    call = dict(zip(l.gene, l.call))
    for d in ("up", "down"):
        both = disc = none = 0
        for a, b in zip(wd.gene_a, wd.gene_b):
            x, y = call.get(a, "unclear"), call.get(b, "unclear")
            if {x, y} <= {d, "nonresp"}:
                if x == y == d: both += 1
                elif x == y == "nonresp": none += 1
                else: disc += 1
        rows.append((s, d, both, disc, none, round(disc / max(both + disc, 1), 3)))
dv = pd.DataFrame(rows, columns=["stress", "direction", "both_respond", "one_responds_other_not",
                                 "neither", "share_discordant_among_responding"])
dv.to_csv(f"{OUT}/wgd_label_divergence.tsv", sep="\t", index=False)
print(dv.to_string(index=False))
