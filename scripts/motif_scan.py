"""Known-motif features for the motif baseline (plan step 5, table 4): FIMO (MEME suite 5.5.9) scan of the
JASPAR 2024 CORE plants non-redundant motifs (805) over the same 3,020-bp windows the models see (CDS masked).

Hits: p < 1e-4, best strand only per position (--max-strand). Each hit is assigned by its centre to one of four
regions of the window: TSS-1000..-501, TSS-500..-1, TSS..TSS+499, TTS half (TTS-499..TTS+1000).
Features = hit counts per motif and region (uint8, capped at 255).

Usage (job-array worker): python -I motif_scan.py scan <project> <windows.fa> <motifs.meme> <i> <n> <work_dir>
         -> <work_dir>/counts_<i>.npz (motif subset i of n)
       python -I motif_scan.py merge <project> <work_dir> <n>
         -> features/motif_jaspar_a4.npy [genes, motifs x 4], .cols.txt, .genes.txt
"""
import gzip, os, subprocess, sys
import numpy as np

MODE, P = sys.argv[1:3]
BIN = os.path.dirname(sys.executable)


def genes_of(fa):
    op = gzip.open if fa.endswith(".gz") else open
    return [l[1:].strip() for l in op(fa, "rt") if l.startswith(">")]


def region(c):                       # 1-based window position of the hit centre -> region 0-3 (spacer -> -1)
    if c <= 500:
        return 0
    if c <= 1000:
        return 1
    if c <= 1500:
        return 2
    return 3 if c > 1520 else -1


if MODE == "scan":
    fa, meme, i, n, work = sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
    motifs = [l.split()[1] for l in open(meme) if l.startswith("MOTIF")]
    sub = motifs[i::n]
    names = genes_of(fa)
    gi = {g: k for k, g in enumerate(names)}
    mi = {m: k for k, m in enumerate(sub)}
    cnt = np.zeros((len(names), len(sub), 4), np.uint16)
    cmd = [os.path.join(BIN, "fimo"), "--text", "--thresh", "1e-4", "--skip-matched-sequence", "--max-strand",
           "--verbosity", "1"]
    for m in sub:
        cmd += ["--motif", m]
    cmd += [meme, fa]
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    nh = 0
    for line in pr.stdout:
        if line.startswith("motif_id") or line.startswith("#"):
            continue
        f = line.split("\t")
        r = region((int(f[3]) + int(f[4])) / 2)
        if r >= 0:
            cnt[gi[f[2]], mi[f[0]], r] += 1
            nh += 1
    assert pr.wait() == 0, "fimo failed"
    os.makedirs(work, exist_ok=True)
    np.savez_compressed(os.path.join(work, f"counts_{i}.npz"), counts=np.minimum(cnt, 255).astype(np.uint8),
                        motifs=np.array(sub))
    print(f"subset {i}/{n}: {len(sub)} motifs, {nh} hits")
else:
    work, n = sys.argv[3], int(sys.argv[4])
    names = genes_of(f"{P}/seqs/wm82a4_windows.fa.gz")
    parts = [np.load(os.path.join(work, f"counts_{i}.npz")) for i in range(n)]
    X = np.concatenate([p["counts"].reshape(len(names), -1) for p in parts], 1)
    cols = [f"{m}|{r}" for p in parts for m in p["motifs"] for r in ("tss_up1000_501", "tss_up500_1", "tss_down0_499", "tts_half")]
    np.save(f"{P}/features/motif_jaspar_a4.npy", X)
    open(f"{P}/features/motif_jaspar_a4.cols.txt", "w").write("\n".join(cols) + "\n")
    open(f"{P}/features/motif_jaspar_a4.genes.txt", "w").write("\n".join(names) + "\n")
    print(f"motif features: {X.shape}, mean hits per gene {X.sum(1).mean():.1f}")
