"""Panel SNPs inside the model windows (plan step 8).

The 1,556-accession panel (Zhang et al. 2022; LIS Wm82.gnm2.div.Zhang_Jiang_2020, 1,511 samples in the VCF) is in
Wm82.a2 coordinates, so the windows are the a2 windows of the same genes (seqs/wm82a2_windows.*, same layout as the
a4 windows: TSS -1000/+500, 20 N, TTS -500/+1000, coding sequence masked). a2 genes are linked to the a4 genes the
models were trained on through the a4 GFF attribute ancestorIdentifier (one-to-one links only).
Every panel site (SoyExpFM/ref/panel/sites_all.by_chrom) inside a window half is placed at its window index
(minus-strand windows are reverse-complemented); sites on masked (N) bases are dropped, and the window base must
equal the reference allele (complemented on the minus strand). Bi-allelic SNPs only.

Usage: python -I variant_sites.py <project> <a4_gff3.gz> <sites_by_chrom_dir>
Writes variants/panel_snps.tsv.gz (chrom, pos, ref, alt, gene_a2, gene_a4, idx, strand) and variants/regions.tsv.
"""
import collections, gzip, os, re, sys
import numpy as np
import pandas as pd

P, GFF, SD = sys.argv[1:4]
os.makedirs(f"{P}/variants", exist_ok=True)
anc = collections.defaultdict(list)
for line in gzip.open(GFF, "rt"):
    f = line.split("\t")
    if len(f) > 8 and f[2] == "gene":
        name = re.search(r"Name=([^;]+)", f[8]).group(1)
        m = re.search(r"ancestorIdentifier=([^;\n]+)", f[8])
        if m:
            for x in m.group(1).split(","):
                anc[x.replace(".Wm82.a2.v1", "").strip()].append(name)
a2_to_a4 = {a2: v[0] for a2, v in anc.items() if len(v) == 1}
a4_count = collections.Counter(a2_to_a4.values())
a2_to_a4 = {a2: a4 for a2, a4 in a2_to_a4.items() if a4_count[a4] == 1}
win = pd.read_csv(f"{P}/seqs/wm82a2_windows.tsv", sep="\t")
seq = {}
for line in gzip.open(f"{P}/seqs/wm82a2_windows.fa.gz", "rt"):
    if line.startswith(">"):
        g = line[1:].strip()
    else:
        seq[g] = line.strip()
win = win[win.gene.isin(a2_to_a4)].reset_index(drop=True)
print(f"a2 windows {len(seq)}; one-to-one a2->a4 links {len(a2_to_a4)}; windows used {len(win)}", flush=True)
comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
out, stats = [], collections.Counter()
for chrom, w in win.groupby("chrom"):
    f = os.path.join(SD, f"{chrom}.tsv.gz")
    if not os.path.exists(f):
        continue
    st = pd.read_csv(f, sep="\t", header=None, names=["chrom", "pos", "alleles"], dtype={"pos": np.int64, "alleles": str})
    pos = st.pos.values
    al = st.alleles.values
    for r in w.itertuples():
        if r.strand == "+":
            halves = [(r.tss - 1000, r.tss + 499, 0, 1), (r.tts - 499, r.tts + 1000, 1520, 1)]
        else:
            halves = [(r.tss - 499, r.tss + 1000, 0, -1), (r.tts - 1000, r.tts + 499, 1520, -1)]
        s = seq[r.gene]
        for lo, hi, off, sgn in halves:
            i0, i1 = np.searchsorted(pos, lo), np.searchsorted(pos, hi, side="right")
            for k in range(i0, i1):
                a = al[k].split(",")
                if len(a) != 2 or len(a[0]) != 1 or len(a[1]) != 1:
                    stats["not_biallelic_snp"] += 1
                    continue
                p = int(pos[k])
                idx = off + (p - lo if sgn == 1 else hi - p)
                base = s[idx]
                if base == "N":
                    stats["masked"] += 1
                    continue
                ref = a[0] if sgn == 1 else comp.get(a[0], "N")
                if base != ref:
                    stats["ref_mismatch"] += 1
                    continue
                out.append((chrom, p, a[0], a[1], r.gene, a2_to_a4[r.gene], idx, r.strand))
                stats["kept"] += 1
    print(chrom, stats["kept"], flush=True)
df = pd.DataFrame(out, columns=["chrom", "pos", "ref", "alt", "gene_a2", "gene_a4", "idx", "strand"])
df.to_csv(f"{P}/variants/panel_snps.tsv.gz", sep="\t", index=False)
reg = df[["chrom", "pos"]].drop_duplicates().sort_values(["chrom", "pos"])
reg.to_csv(f"{P}/variants/regions.tsv", sep="\t", index=False, header=False)
print(dict(stats), f"unique sites {len(reg)}; genes {df.gene_a4.nunique()}")
