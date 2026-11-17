"""Summary numbers quoted in the Results that are not a column of another result file.

  expression context, per stress (labels: cre_data.strict_calls; gene features: features/gene_covariates.tsv):
      median tau and median breadth (share of atlas runs with TPM > 1) of responsive (up or down), up-regulated,
      down-regulated and non-responsive genes;
  ABRE sites in the TTS half of the windows of up-regulated genes (results/abre_contrib_position.tsv): share of all ABRE
      sites and share of the summed net contribution of ABRE sites;
  similarity check (results/eval_v1_cand3_vs_stack[_nosim]_*.tsv): share of the labelled test genes of each of the nine
      main tasks removed because they share a window segment of >= 300 bp at >= 90% identity with a training gene.
Usage: python -I text_numbers.py <project>    -> results/text_numbers.tsv (quantity, key, value)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_data import strict_calls   # noqa: E402

P = sys.argv[1]
rows = []
cov = pd.read_csv(f"{P}/features/gene_covariates.tsv", sep="\t").set_index("gene")
lab = strict_calls(pd.read_csv(f"{P}/labels/labels_v1.tsv.gz", sep="\t"))
for s in ["drought", "salt", "heat", "cold", "alkaline"]:
    q = lab[lab.stress == s]
    for name, genes in (("responsive", q.gene[q.call.isin(["up", "down"])]), ("up", q.gene[q.call == "up"]),
                        ("down", q.gene[q.call == "down"]), ("nonresp", q.gene[q.call == "nonresp"])):
        c = cov.reindex(genes)
        rows.append(("median tau", f"{s} {name}", round(float(c.expr_tau.median()), 4)))
        rows.append(("median breadth", f"{s} {name}", round(float(c.expr_breadth.median()), 4)))
pos = pd.read_csv(f"{P}/results/abre_contrib_position.tsv", sep="\t")
pos["total"] = pos.sites * pos.mean_net_contribution
for t, g in pos[pos.task.str.endswith("-up")].groupby("task"):
    tts = g[g.half == "TTS"]
    rows.append(("ABRE sites in the TTS half, share of sites", t, round(float(tts.sites.sum() / g.sites.sum()), 4)))
    rows.append(("ABRE sites in the TTS half, share of summed net contribution", t, round(float(tts.total.sum() / g.total.sum()), 4)))
MAIN9 = [f"{s}-{d}" for s in ["drought", "salt", "heat", "cold", "alkaline"] for d in ("up", "down")][:-1]
for d, ref in [("main_seed0", "seed0"), ("main_seed1", "seed1"), ("main_seed2", "seed2"), ("chrom", "chrom")]:
    a = pd.read_csv(f"{P}/results/eval_v1_cand3_vs_stack_nosim_{d}.tsv", sep="\t").set_index("task").n.reindex(MAIN9)
    b = pd.read_csv(f"{P}/results/eval_v1_cand3_vs_stack_{ref}.tsv", sep="\t").set_index("task").n.reindex(MAIN9)
    r = 1 - a / b
    rows.append(("similarity check, share of labelled test genes removed, min over tasks", d, round(float(r.min()), 4)))
    rows.append(("similarity check, share of labelled test genes removed, max over tasks", d, round(float(r.max()), 4)))
out = pd.DataFrame(rows, columns=["quantity", "key", "value"])
out.to_csv(f"{P}/results/text_numbers.tsv", sep="\t", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(out.to_string(index=False))
