"""Extract one 3,020-bp model input per gene (DeepCRE layout), with every annotated CDS masked to N.

Window = TSS-1000 .. TSS+499 (1,500 bp; TSS is the 1,001st base)  +  20 N  +
         TTS-499 .. TTS+1000 (1,500 bp; TTS is the 500th base of this half)
in transcript orientation (minus-strand genes are reverse-complemented). TSS/TTS are the ends of the
primary transcript (GFF mRNA with longest=1; the longest mRNA if none is flagged). All CDS of all genes
and isoforms that fall in a window are replaced by N, so the model cannot recognise gene families from
coding sequence. Positions beyond chromosome ends are N.

Usage: python3 -I make_windows.py <genome.fa[.gz]> <genes.gff3[.gz]> <out_prefix>
Writes <out_prefix>.fa.gz and <out_prefix>.tsv (gene, chrom, strand, tss, tts, has_5utr, has_3utr,
cds_masked_bp, n_bp).
"""
import collections, gzip, sys
import numpy as np

FA, GFF, OUT = sys.argv[1:4]
UP, DOWN, SPACER = 1000, 500, 20
COMP = bytes.maketrans(b"ACGTNacgtn", b"TGCANtgcan")


def opener(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def read_genome(path):
    seqs, name, buf = {}, None, []
    for line in opener(path):
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf).upper().encode()
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name:
        seqs[name] = "".join(buf).upper().encode()
    return seqs


def attrs(s):
    return dict(kv.split("=", 1) for kv in s.strip().split(";") if "=" in kv)


genes = {}                                    # gene id -> (chrom, strand, name)
mrnas = collections.defaultdict(list)         # gene id -> [(longest_flag, length, mrna id, start, end)]
cds = collections.defaultdict(list)           # chrom -> [(start0, end0)]
utr5, utr3 = set(), set()
for line in opener(GFF):
    if line.startswith("#"):
        continue
    f = line.rstrip("\n").split("\t")
    if len(f) < 9:
        continue
    a = attrs(f[8])
    t = f[2]
    if t == "gene":
        genes[a["ID"]] = (f[0], f[6], a.get("Name", a["ID"]))
    elif t == "mRNA":
        s, e = int(f[3]), int(f[4])
        mrnas[a["Parent"]].append((a.get("longest") == "1", e - s + 1, a["ID"], s, e))
    elif t == "CDS":
        cds[f[0]].append((int(f[3]) - 1, int(f[4])))
    elif t == "five_prime_UTR":
        utr5.add(a["Parent"])
    elif t == "three_prime_UTR":
        utr3.add(a["Parent"])

genome = read_genome(FA)
by_chrom = collections.defaultdict(list)
for gid, (c, strand, name) in genes.items():
    if not mrnas.get(gid):
        continue
    flag, length, mid, s, e = sorted(mrnas[gid], key=lambda x: (x[0], x[1]), reverse=True)[0]
    by_chrom[c].append((name, strand, s, e, mid))

n_out = 0
with gzip.open(OUT + ".fa.gz", "wt", compresslevel=6) as fo, open(OUT + ".tsv", "w") as ft:
    ft.write("gene\tchrom\tstrand\ttss\ttts\thas_5utr\thas_3utr\tcds_masked_bp\tn_bp\n")
    for c in sorted(by_chrom):
        if c not in genome:
            print("missing chromosome in genome:", c, file=sys.stderr)
            continue
        raw = np.frombuffer(genome[c], dtype=np.uint8).copy()
        L = len(raw)
        mask = np.zeros(L, dtype=bool)
        for s0, e0 in cds.get(c, []):
            mask[s0:e0] = True
        masked = raw.copy()
        masked[mask] = ord("N")

        def grab(s1, e1):
            """1-based inclusive genomic interval -> bytes, N beyond the ends."""
            out = np.full(e1 - s1 + 1, ord("N"), dtype=np.uint8)
            lo, hi = max(s1, 1), min(e1, L)
            if lo <= hi:
                out[lo - s1: hi - s1 + 1] = masked[lo - 1: hi]
            return out.tobytes()

        def nmask(s1, e1):
            lo, hi = max(s1, 1), min(e1, L)
            return int(mask[lo - 1: hi].sum()) if lo <= hi else 0

        for name, strand, s, e, mid in sorted(by_chrom[c], key=lambda x: x[2]):
            if strand == "+":
                tss, tts = s, e
                a1, a2 = tss - UP, tss + DOWN - 1
                b1, b2 = tts - DOWN + 1, tts + UP
                seq = grab(a1, a2) + b"N" * SPACER + grab(b1, b2)
            else:
                tss, tts = e, s
                a1, a2 = tss - DOWN + 1, tss + UP
                b1, b2 = tts - UP, tts + DOWN - 1
                seq = (grab(a1, a2).translate(COMP)[::-1] + b"N" * SPACER +
                       grab(b1, b2).translate(COMP)[::-1])
            assert len(seq) == 2 * (UP + DOWN) + SPACER
            m = nmask(a1, a2) + nmask(b1, b2)
            ft.write(f"{name}\t{c}\t{strand}\t{tss}\t{tts}\t{int(mid in utr5)}\t{int(mid in utr3)}\t{m}\t{seq.count(b'N')}\n")
            fo.write(f">{name}\n{seq.decode()}\n")
            n_out += 1
print(f"{n_out} windows -> {OUT}.fa.gz / .tsv")
