"""Change in predicted log odds for the in-silico insertions (adds two columns to the output of insertion.py).

p0  = StressCRE's predicted probability for the unmodified window of the background gene and task (mean over the three
      models that had the gene in their test set: results/v1_stresscre_allgenes.tsv.gz, the same models insertion.py uses)
lor = logit(p0 + effect) - logit(p0), where effect is the change in probability written by insertion.py.
Existing p0/lor columns are recomputed, so the script can be run again after insertion.py adds constructs.

Usage: python -I insertion_logodds.py <project> [insertion file, default results/insertion_effects.tsv.gz]
"""
import sys
import numpy as np
import pandas as pd

P = sys.argv[1]
f = f"{P}/{sys.argv[2] if len(sys.argv) > 2 else 'results/insertion_effects.tsv.gz'}"
ins = pd.read_csv(f, sep="\t").drop(columns=["p0", "lor"], errors="ignore")
base = pd.read_csv(f"{P}/results/v1_stresscre_allgenes.tsv.gz", sep="\t")[["gene", "task", "p"]]
ins = ins.merge(base.rename(columns={"gene": "background", "p": "p0"}), on=["background", "task"], how="left")
assert ins.p0.notna().all(), "background gene without a StressCRE prediction"
logit = lambda p: np.log(p / (1 - p))
ins["lor"] = logit(ins.p0 + ins.effect) - logit(ins.p0)
ins.to_csv(f, sep="\t", index=False)
print(f"{f}: {len(ins)} rows; p0 and lor added")
