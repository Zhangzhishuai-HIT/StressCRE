"""Count usable stress-vs-control contrasts per stress, now (quantified runs) and when the current
SoyExpFM round is finished (all annotated runs). Writes results/contrast_inventory.tsv.

Usage: python3 -I inventory_contrasts.py <project_dir>
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contrasts import ABIO, SHORT, load_annotation, find_contrasts  # noqa: E402

P = sys.argv[1]
a = load_annotation()
rows = []
for s in ABIO + ["iron deficiency"]:
    for mode, qonly in (("all annotated runs", False), ("quantified now", True)):
        for m in (2, 1):
            c = find_contrasts(a, s, min_per_arm=m, quantified_only=qonly)
            df = pd.DataFrame(c)
            if df.empty:
                rows.append((SHORT[s], mode, m, 0, 0, 0, 0, 0, "")); continue
            nrun = sum(len(x) + len(y) for x, y in zip(df.treated, df.control))
            gxe = int((df.groupby("study").cultivar.nunique() >= 2).sum())
            soja = len(find_contrasts(a, s, min_per_arm=m, quantified_only=qonly, species="Glycine soja"))
            org = df.organ_group.value_counts()
            rows.append((SHORT[s], mode, m, df.study.nunique(), len(df), nrun, gxe, soja,
                         ", ".join(f"{k} {v}" for k, v in org.items())))
out = pd.DataFrame(rows, columns=["stress", "runs_used", "min_per_arm", "studies", "contrasts", "runs",
                                  "studies_with_2plus_cultivars", "soja_contrasts", "organ_groups"])
os.makedirs(f"{P}/results", exist_ok=True)
out.to_csv(f"{P}/results/contrast_inventory.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
print(out.to_string(index=False))
