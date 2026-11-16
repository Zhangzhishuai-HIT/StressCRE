"""k-mer composition features for the k-mer baseline (plan step 5).

For each 3,020-bp window: canonical k-mer frequencies, k = 1..6, counted separately in the TSS half
(first 1,500 bp) and the TTS half (last 1,500 bp); k-mers containing N (masked CDS, padding) are
skipped and counts are divided by the number of valid k-mers. 2 x 2,772 = 5,544 features.
No feature selection here, so nothing is learned from the labels.

Usage: python3 -I kmer_features.py <windows.fa.gz> <out_prefix>   -> <out_prefix>.npy (float32), .genes.txt, .cols.txt
"""
import gzip, itertools, sys
import numpy as np

FA, OUT = sys.argv[1:3]
KS = range(1, 7)
HALF = 1500
LUT = np.full(256, -1, dtype=np.int8)
for i, c in enumerate(b"ACGT"):
    LUT[c] = i


def canon_map(k):
    """index of every k-mer code -> index of its canonical (min of kmer, revcomp) among canonical k-mers."""
    codes = np.arange(4 ** k)
    digits = np.stack([(codes >> (2 * (k - 1 - j))) & 3 for j in range(k)], axis=1)
    rc = (3 - digits[:, ::-1])
    rc_code = (rc * (4 ** np.arange(k - 1, -1, -1))).sum(1)
    can = np.minimum(codes, rc_code)
    uniq = np.unique(can)
    pos = {c: i for i, c in enumerate(uniq)}
    names = ["".join("ACGT"[d] for d in digits[c]) for c in uniq]
    return np.array([pos[c] for c in can]), len(uniq), names


maps = {k: canon_map(k) for k in KS}

names, seqs = [], []
name = None
for line in gzip.open(FA, "rt"):
    if line.startswith(">"):
        name = line[1:].strip()
    else:
        names.append(name); seqs.append(line.strip().encode())
X = np.frombuffer(b"".join(seqs), dtype=np.uint8).reshape(len(seqs), -1)
L = X.shape[1]
enc = LUT[X]                                    # -1 for N
halves = {"tss": enc[:, :HALF], "tts": enc[:, L - HALF:]}

cols, blocks = [], []
CH = 4000
for hname, H in halves.items():
    for k in KS:
        cmap, ncan, kn = maps[k]
        out = np.zeros((H.shape[0], ncan), dtype=np.float32)
        for s in range(0, H.shape[0], CH):
            h = H[s:s + CH].astype(np.int32)
            n, w = h.shape
            code = np.zeros((n, w - k + 1), dtype=np.int64)
            bad = np.zeros((n, w - k + 1), dtype=bool)
            for j in range(k):
                part = h[:, j:w - k + 1 + j]
                code = code * 4 + np.where(part < 0, 0, part)
                bad |= part < 0
            code = cmap[code]
            code = np.where(bad, ncan, code)                      # junk bin for N-containing k-mers
            flat = (code + (np.arange(n)[:, None] * (ncan + 1))).ravel()
            cnt = np.bincount(flat, minlength=n * (ncan + 1)).reshape(n, ncan + 1)[:, :ncan]
            tot = cnt.sum(1, keepdims=True)
            out[s:s + n] = cnt / np.maximum(tot, 1)
        blocks.append(out)
        cols += [f"{hname}_{m}" for m in kn]
    print(hname, "done", flush=True)
F = np.concatenate(blocks, axis=1)
np.save(OUT + ".npy", F)
open(OUT + ".genes.txt", "w").write("\n".join(names) + "\n")
open(OUT + ".cols.txt", "w").write("\n".join(cols) + "\n")
print(F.shape, "features ->", OUT + ".npy")
