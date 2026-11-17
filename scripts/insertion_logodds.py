"""Merge the in-silico insertion runs and add the change in predicted log odds.

insertion.py writes one file per run (each run overwrites its --out file). The paper used two runs (jobs/insertion_runs.sh):
the main run (results/insertion_effects.tsv.gz) and the singles at -132 bp (results/insertion_m132.tsv.gz, the position
of the second site in pairs that begin with an ABRE). --add appends further run files to the main file; rows already
present (same background, construct and task) are kept once, so the script can be run again.
Then, for every row:
  p0  = StressCRE's predicted probability for the unmodified window of the background gene and task (mean over the three
        models that had the gene in their test set: results/v1_stresscre_allgenes.tsv.gz, the models insertion.py uses)
  lor = logit(p0 + effect) - logit(p0), where effect is the change in probability written by insertion.py.

Usage: python -I insertion_logodds.py <project> [--add results/insertion_m132.tsv.gz ...]
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("project"); ap.add_argument("--main", default="results/insertion_effects.tsv.gz")
ap.add_argument("--add", nargs="*", default=[])
a = ap.parse_args()
P = a.project
f = f"{P}/{a.main}"
ins = pd.concat([pd.read_csv(x, sep="\t") for x in [f] + [f"{P}/{y}" for y in a.add]], ignore_index=True)
ins = ins.drop(columns=["p0", "lor"], errors="ignore").drop_duplicates(["background", "construct", "task"], keep="first")
base = pd.read_csv(f"{P}/results/v1_stresscre_allgenes.tsv.gz", sep="\t")[["gene", "task", "p"]]
ins = ins.merge(base.rename(columns={"gene": "background", "p": "p0"}), on=["background", "task"], how="left")
assert ins.p0.notna().all(), "background gene without a StressCRE prediction"
logit = lambda p: np.log(p / (1 - p))
ins["lor"] = logit(ins.p0 + ins.effect) - logit(ins.p0)
ins.to_csv(f, sep="\t", index=False)
print(f"{f}: {len(ins)} rows; p0 and lor added")
