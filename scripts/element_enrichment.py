"""Enrichment of known stress elements near the TSS of responsive genes (data side of plan step 6).

For each task (stress x direction), the share of responsive and of non-responsive genes with at least one site of an
element (either strand) within 500 bp of the TSS (window positions 500-1499, unmasked bases) and the odds ratio with
Fisher's exact test. Elements: ABRE ACGTG(G|T), G-box CACGTG, DRE/CRT (A|G)CCGAC, HSE (three units)
GAAnnTTCnnGAA, W-box TTGAC(C|T), GCC-box AGCCGCC, evening element AAAATATCT.
Usage: python -I element_enrichment.py <project>    -> results/element_enrichment.tsv
"""
import gzip, os, re, sys
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_data import strict_calls   # noqa: E402

P = sys.argv[1]
ELEM = {"ABRE": "ACGTG[GT]", "G-box": "CACGTG", "DRE/CRT": "[AG]CCGAC", "HSE": "GAA..TTC..GAA|TTC..GAA..TTC",
        "W-box": "TTGAC[CT]", "GCC-box": "AGCCGCC", "EE": "AAAATATCT"}
COMP = str.maketrans("ACGTN", "TGCAN")
seqs = {}
for line in gzip.open(f"{P}/seqs/wm82a4_windows.fa.gz", "rt"):
    if line.startswith(">"):
        g = line[1:].strip()
    else:
        seqs[g] = line.strip()[500:1500]
pres = {k: {g: bool(re.search(v, s)) or bool(re.search(v, s.translate(COMP)[::-1])) for g, s in seqs.items()} for k, v in ELEM.items()}
lab = strict_calls(pd.read_csv(f"{P}/labels/labels_v1.tsv.gz", sep="\t"))
rows = []
for s in ["drought", "salt", "heat", "cold", "alkaline"]:
    q = lab[(lab.stress == s) & lab.gene.isin(seqs)]
    nr = q[q.call == "nonresp"].gene
    for d in ("up", "down"):
        r = q[q.call == d].gene
        for k in ELEM:
            a, b = sum(pres[k][g] for g in r), sum(pres[k][g] for g in nr)
            o, p = fisher_exact([[a, len(r) - a], [b, len(nr) - b]])
            rows.append(dict(task=f"{s}-{d}", element=k, n_resp=len(r), share_resp=round(a / len(r), 4),
                             share_nonresp=round(b / len(nr), 4), odds_ratio=round(o, 3), p=float(f"{p:.3g}")))
out = pd.DataFrame(rows)
from statsmodels.stats.multitest import multipletests   # noqa: E402
out["q"] = [float(f"{q:.3g}") for q in multipletests(out.p, method="fdr_bh")[1]]   # Benjamini-Hochberg over all element x task tests
out.to_csv(f"{P}/results/element_enrichment.tsv", sep="\t", index=False)
pd.set_option("display.width", 200)
print(out.pivot(index="element", columns="task", values="odds_ratio").round(2).to_string())
