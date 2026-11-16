"""Check the 3,020-bp windows against the transcript sequences (orientation, TSS/TTS position, masking).

For each gene whose primary transcript has UTRs, the window bases right after the TSS (window index
1000..) must equal the first transcript bases, and the bases ending at the TTS (index 1520+499) must
equal the last transcript bases, wherever the window base is not N (CDS is masked). Also prints a
20-gene listing (10 per strand, seed 0) for the record and summary statistics.

Usage: python3 -I check_windows.py <windows.fa.gz> <windows.tsv> <transcripts.fa> <out_tsv>
"""
import gzip, random, sys
import numpy as np
import pandas as pd

WFA, WTSV, TFA, OUT = sys.argv[1:5]
K = 30


def read_fa(path):
    op = gzip.open if path.endswith(".gz") else open
    seqs, name, buf = {}, None, []
    for line in op(path, "rt"):
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf).upper()
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name:
        seqs[name] = "".join(buf).upper()
    return seqs


win = read_fa(WFA)
tab = pd.read_csv(WTSV, sep="\t")
tx = read_fa(TFA)
# primary transcript id = gene + ".1" is not guaranteed; map gene -> transcripts present
by_gene = {}
for t in tx:
    g = t.split("ann1.")[-1].rsplit(".", 1)[0]
    by_gene.setdefault(g, []).append(t)


def agree(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x != "N"]
    return (sum(x == y for x, y in pairs), len(pairs))


rows = []
for r in tab.itertuples():
    w = win[r.gene]
    best = None
    for t in by_gene.get(r.gene, []):
        s = tx[t]
        m5, n5 = agree(w[1000:1000 + K], s[:K])
        m3, n3 = agree(w[1520 + 500 - K:1520 + 500], s[-K:])
        score = (m5 + m3, n5 + n3)
        if best is None or score[0] > best[0][0]:
            best = (score, t, m5, n5, m3, n3, len(s))
    if best is None:
        rows.append((r.gene, r.strand, r.has_5utr, r.has_3utr, "", 0, 0, 0, 0, r.cds_masked_bp)); continue
    _, t, m5, n5, m3, n3, L = best
    rows.append((r.gene, r.strand, r.has_5utr, r.has_3utr, t, m5, n5, m3, n3, r.cds_masked_bp))
d = pd.DataFrame(rows, columns=["gene", "strand", "has_5utr", "has_3utr", "transcript", "tss_match", "tss_compared",
                                "tts_match", "tts_compared", "cds_masked_bp"])
d.to_csv(OUT, sep="\t", index=False)
u5 = d[(d.has_5utr == 1) & (d.tss_compared > 0)]
u3 = d[(d.has_3utr == 1) & (d.tts_compared > 0)]
print(f"genes {len(d)}; with a transcript {int((d.transcript != '').sum())}")
print(f"TSS side (genes with 5'UTR, {len(u5)}): unmasked bases matching transcript start "
      f"{u5.tss_match.sum() / u5.tss_compared.sum():.4f}; genes with every compared base matching "
      f"{(u5.tss_match == u5.tss_compared).mean():.4f}")
print(f"TTS side (genes with 3'UTR, {len(u3)}): {u3.tts_match.sum() / u3.tts_compared.sum():.4f}; all-match genes "
      f"{(u3.tts_match == u3.tts_compared).mean():.4f}")
for s in "+-":
    x = u5[u5.strand == s]
    print(f"  strand {s}: TSS all-match {(x.tss_match == x.tss_compared).mean():.4f} (n={len(x)})")
q = np.percentile(d.cds_masked_bp, [10, 25, 50, 75, 90])
print("CDS-masked bp per window, 10/25/50/75/90%:", q.astype(int).tolist(),
      "| windows with any masking:", f"{(d.cds_masked_bp > 0).mean():.3f}")
random.seed(0)
pick = random.sample(list(d[d.strand == "+"].gene), 10) + random.sample(list(d[d.strand == "-"].gene), 10)
print(d[d.gene.isin(pick)].to_string(index=False))
