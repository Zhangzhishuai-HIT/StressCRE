"""Motifs learned by StressCRE (plan step 6): tfmodisco-lite patterns of every task, their matches to known motifs
and their positions in the window.

For each pattern (positive and negative metaclusters of work/modisco/<task>/modisco.h5): the contribution weight
matrix (CWM) is trimmed to the positions whose summed |CWM| reaches 30% of its maximum, and patterns still wider than
20 bp (diffuse, mostly compositional) to the 20-bp window with the largest summed |CWM|; the frequency matrix of the
trimmed positions is written in MEME format and compared with Tomtom (MEME suite 5.5.9; Pearson distance, min
overlap 5) to JASPAR 2024 CORE plants and to the soybean DAP-seq library (features/soy_dapseq_streme_c100.meme);
the best match with q < 0.05 is kept. The share of sites whose trimmed positions contain an exact ACGTG core (either
strand) is recorded. Seqlet centres are given relative to the TSS (window positions 0-1499,
TSS = position 1000) or the TTS (positions 1520-3019, TTS = position 2019).

Usage: python -I modisco_motifs.py <project>
Writes work/modisco/patterns.meme, results/modisco_patterns.tsv, results/modisco_seqlets.tsv.gz.
"""
import os, re, subprocess, sys
import h5py
import numpy as np
import pandas as pd

P = sys.argv[1]
BIN = os.path.dirname(sys.executable)
TASKS = [f"{s}-{d}" for s in ["drought", "salt", "heat", "cold", "alkaline"] for d in ("up", "down")]
J = "/home/user/zhangzhishuai/myhermes/down_data/JASPAR2024/JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt"
DAP = f"{P}/features/soy_dapseq_streme_c100.meme"
jname = {l.split()[1]: l.split()[2] for l in open(J) if l.startswith("MOTIF")}
MAXW = 20
rows, seql, meme = [], [], ["MEME version 5\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.3 C 0.2 G 0.2 T 0.3\n\n"]
for t in TASKS:
    f = f"{P}/work/modisco/{t}/modisco.h5"
    genes = open(f"{P}/work/modisco/{t}/genes.txt").read().split()
    h = h5py.File(f, "r")
    for mc in ("pos_patterns", "neg_patterns"):
        if mc not in h:
            continue
        for k in sorted(h[mc], key=lambda x: int(x.split("_")[1])):
            g = h[mc][k]
            cwm, pwm = g["contrib_scores"][:], g["sequence"][:]
            sc = np.abs(cwm).sum(1)
            keep = np.where(sc >= 0.3 * sc.max())[0]
            lo, hi = keep.min(), keep.max() + 1
            if hi - lo > MAXW:                     # diffuse patterns: keep the MAXW-bp core with the largest |CWM|
                cs = np.convolve(sc, np.ones(MAXW), "valid")
                lo = int(cs.argmax()); hi = lo + MAXW
            pw = pwm[lo:hi] / pwm[lo:hi].sum(1, keepdims=True)
            cons = "".join("ACGT"[i] if p.max() >= 0.5 else "acgt"[i] for i, p in zip(pw.argmax(1), pw))
            name = f"{t}|{mc[:3]}|{k.split('_')[1]}"
            s = g["seqlets"]
            n = len(s["start"])
            sq = s["sequence"][:, lo:hi]                   # aligned one-hot sites; masked bases are all zero (N)
            txt = np.where(sq.max(2) > 0, np.array(list("ACGT"))[sq.argmax(2)], "N")
            core = np.mean([bool(re.search("ACGTG|CACGT", "".join(r))) for r in txt])   # exact ACGTG core, either strand
            ctr = (s["start"][:] + s["end"][:]) / 2
            rel = np.where(ctr < 1500, ctr - 1000, np.where(ctr >= 1520, ctr - 2019, np.nan))
            in_tss = ctr < 1500
            rows.append(dict(task=t, metacluster=mc[:3], pattern=int(k.split("_")[1]), name=name, n_seqlets=n,
                             n_genes=len(set(s["example_idx"][:])), consensus=cons, width=hi - lo,
                             cwm_sum=round(float(cwm[lo:hi].sum()), 4), share_acgtg_core=round(float(core), 3),
                             frac_tss_half=round(float(in_tss.mean()), 3),
                             median_pos_tss_half=float(np.nanmedian(rel[in_tss])) if in_tss.any() else np.nan,
                             median_pos_tts_half=float(np.nanmedian(rel[~in_tss & (ctr >= 1520)])) if (ctr >= 1520).any() else np.nan))
            for e, c, rcf in zip(s["example_idx"][:], ctr, s["is_revcomp"][:]):
                seql.append((t, name, genes[int(e)], float(c), bool(rcf)))
            meme.append(f"MOTIF {name.replace('|', '_')} {cons}\nletter-probability matrix: alength= 4 w= {hi - lo} nsites= {n} E= 0\n"
                        + "".join(" ".join(f"{x:.4f}" for x in r) + "\n" for r in pw) + "\n")
mf = f"{P}/work/modisco/patterns.meme"
open(mf, "w").write("".join(meme))
pat = pd.DataFrame(rows)
for db, tag in ((J, "jaspar"), (DAP, "dapseq")):
    r = subprocess.run([os.path.join(BIN, "tomtom"), "-no-ssc", "-text", "-min-overlap", "5", "-dist", "pearson",
                        "-evalue", "-thresh", "10", mf, db], capture_output=True, text=True, check=True)
    tt = pd.read_csv(pd.io.common.StringIO(r.stdout), sep="\t", comment="#")
    tt = tt[tt["q-value"] < 0.05].sort_values("q-value").drop_duplicates("Query_ID")
    tt["name"] = tt.Query_ID.str.replace("_pos_", "|pos|").str.replace("_neg_", "|neg|")
    lab = tt.Target_ID.map(jname) if tag == "jaspar" else tt.Target_ID
    m = dict(zip(tt.name, lab + " (" + tt.Target_ID + ")" if tag == "jaspar" else lab))
    q = dict(zip(tt.name, tt["q-value"]))
    pat[f"{tag}_match"] = pat.name.map(m)
    pat[f"{tag}_q"] = pat.name.map(q)
pat.to_csv(f"{P}/results/modisco_patterns.tsv", sep="\t", index=False)
pd.DataFrame(seql, columns=["task", "pattern", "gene", "center_idx", "revcomp"]).to_csv(f"{P}/results/modisco_seqlets.tsv.gz", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
for t in TASKS:
    print(pat[(pat.task == t) & (pat.metacluster == "pos")].head(6)[["name", "n_seqlets", "consensus", "frac_tss_half", "median_pos_tss_half", "jaspar_match", "dapseq_match"]].to_string(index=False, header=(t == TASKS[0])))
