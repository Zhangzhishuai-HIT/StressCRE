"""One-off repair of the PlantCAD2 chunks (2026-11-16).

The AutoDL run of embed_plantcad2.py (batch 16, chunk 1000) did not clip the last batch of a chunk at the chunk
end, so chunks 0-51 held 1,008 rows: the 1,000 genes listed in chunk_XXXX.genes.txt, then the first 8 genes of the
next chunk. This keeps the first len(genes.txt) rows of each chunk after checking that the extra rows equal the
next chunk's first rows (r > 0.9999), then checks that the chunks together list every window once, in FASTA order.

Usage: python -I fix_embed_chunk_rows.py <emb_dir> <windows.fa.gz>
"""
import glob, gzip, os, sys
import numpy as np

D, FA = sys.argv[1:3]
ks = sorted(int(os.path.basename(f)[6:10]) for f in glob.glob(os.path.join(D, "chunk_????.npy")))
assert ks == list(range(len(ks))), "missing chunks"
names = []
for k in ks:
    f = os.path.join(D, f"chunk_{k:04d}.npy")
    gn = open(os.path.join(D, f"chunk_{k:04d}.genes.txt")).read().split()
    a = np.load(f, mmap_mode="r")
    if a.shape[0] > len(gn):
        ex = np.asarray(a[len(gn):], dtype=np.float32)
        nxt = np.load(os.path.join(D, f"chunk_{k + 1:04d}.npy"), mmap_mode="r")
        ref = np.asarray(nxt[:len(ex)], dtype=np.float32)
        r = min(np.corrcoef(ex[i].ravel(), ref[i].ravel())[0, 1] for i in range(len(ex)))
        assert r > 0.9999, (k, r)
        np.save(f + ".tmp.npy", np.ascontiguousarray(a[:len(gn)]))
        del a, nxt
        os.replace(f + ".tmp.npy", f)
        print(f"chunk {k}: {len(gn) + len(ex)} -> {len(gn)} rows (extra rows = next chunk's first {len(ex)}, min r {r:.6f})")
    else:
        assert a.shape[0] == len(gn), (k, a.shape, len(gn))
        print(f"chunk {k}: {len(gn)} rows, unchanged")
    names += gn
fa = [l[1:].strip() for l in gzip.open(FA, "rt") if l.startswith(">")]
assert names == fa, "gene order differs from the FASTA"
print(f"OK: {len(ks)} chunks, {len(names)} genes, same order as {os.path.basename(FA)}")
