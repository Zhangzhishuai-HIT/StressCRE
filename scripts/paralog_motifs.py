"""Known stress elements in the two copies of divergent WGD pairs (plan step 7, descriptive part).

For every task, divergent pairs = one copy strictly up (or down), the other non-responsive (pairs from
results/wgd_pairs.tsv with protein identity < 99%). In both copies' windows, sites of the canonical element of the
stress are counted on both strands (regular expressions on the unmasked bases of the TSS half, positions
-1000..+500, and of the TTS half): ABRE ACGTG(G|T) for drought and salt, DRE/CRT CCGAC for cold (and drought),
HSE nGAAnnTTCn for heat, W-box TTGAC(C|T) and G-box CACGTG as further references.
Reported per task x element: pairs, share of pairs in which the responding copy has more sites (among pairs whose
counts differ), sign-test P, and the same for all pairs whose two copies are both labelled, against the measured
log2 fold-change difference (Spearman).
Usage: python -I paralog_motifs.py <project>    -> results/paralog_motifs.tsv
"""
import gzip, os, re, sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_data import strict_calls   # noqa: E402

P = sys.argv[1]
ELEM = {"ABRE": ("ACGTG[GT]", 6), "DRE/CRT": ("CCGAC", 5), "HSE": ("GAA..TTC", 8), "W-box": ("TTGAC[CT]", 6), "G-box": ("CACGTG", 6)}
COMP = str.maketrans("ACGTN", "TGCAN")
PAT = {k: re.compile("(?=" + v[0] + ")") for k, v in ELEM.items()}


def count(seq, k):
    """sites on either strand, counted once (palindromic sites match both strands at the same place)"""
    w, L = ELEM[k][1], len(seq)
    f = {m.start() for m in PAT[k].finditer(seq)}
    r = {L - m.start() - w for m in PAT[k].finditer(seq.translate(COMP)[::-1])}
    return len(f | r)


seqs = {}
for line in gzip.open(f"{P}/seqs/wm82a4_windows.fa.gz", "rt"):
    if line.startswith(">"):
        g = line[1:].strip()
    else:
        seqs[g] = line.strip()
pairs = pd.read_csv(f"{P}/results/wgd_pairs.tsv", sep="\t")
pairs = pairs[(pairs.protein_identity < 99) & pairs.gene_a.isin(seqs) & pairs.gene_b.isin(seqs)].reset_index(drop=True)
lab = strict_calls(pd.read_csv(f"{P}/labels/labels_v1.tsv.gz", sep="\t"))
call = lab.set_index(["gene", "stress"]).call
lfc = lab.set_index(["gene", "stress"]).meta_lfc
genes = sorted(set(pairs.gene_a) | set(pairs.gene_b))
cnt = {(g, k, h): count(seqs[g][:1500] if h == "tss" else seqs[g][1520:], k) for g in genes for k in ELEM for h in ("tss", "tts")}
rows = []
for s in ["drought", "salt", "heat", "cold", "alkaline"]:
    for d in ("up", "down"):
        ca = np.array([call.get((g, s), "none") for g in pairs.gene_a])
        cb = np.array([call.get((g, s), "none") for g in pairs.gene_b])
        both = np.isin(ca, [d, "nonresp"]) & np.isin(cb, [d, "nonresp"])
        div = both & (((ca == d) & (cb == "nonresp")) | ((ca == "nonresp") & (cb == d)))
        sign = np.where(ca == d, 1, -1)
        la = np.array([lfc.get((g, s), np.nan) for g in pairs.gene_a])
        lb = np.array([lfc.get((g, s), np.nan) for g in pairs.gene_b])
        meas = (la - lb) * (1 if d == "up" else -1)
        for k in ELEM:
            for h in ("tss", "tts"):
                na = np.array([cnt[(g, k, h)] for g in pairs.gene_a])
                nb = np.array([cnt[(g, k, h)] for g in pairs.gene_b])
                dd = (na - nb) * sign
                dv = dd[div]
                nz = dv[dv != 0]
                more = int((nz > 0).sum())
                p = binomtest(more, len(nz), 0.5).pvalue if len(nz) else np.nan
                ok = both & ~np.isnan(meas)
                r = spearmanr(na[ok] - nb[ok], meas[ok]).correlation if ok.sum() > 10 else np.nan
                rows.append(dict(task=f"{s}-{d}", element=k, half=h, divergent_pairs=int(div.sum()), pairs_differing=len(nz),
                                 responder_more=more, share_responder_more=round(more / len(nz), 3) if len(nz) else np.nan,
                                 sign_test_p=float(f"{p:.3g}") if len(nz) else np.nan, labelled_pairs=int(ok.sum()),
                                 spearman_count_diff_vs_lfc_diff=round(r, 4) if r == r else np.nan))
out = pd.DataFrame(rows)
out.to_csv(f"{P}/results/paralog_motifs.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
key = out[((out.task.isin(["drought-up", "salt-up"])) & (out.element == "ABRE")) | ((out.task == "cold-up") & (out.element == "DRE/CRT")) |
          ((out.task == "heat-up") & (out.element == "HSE"))]
print(key.to_string(index=False))
