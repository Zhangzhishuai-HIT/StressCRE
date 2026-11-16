"""Write one count matrix per stress-vs-control contrast (quantified runs, >= 2 per arm).

Output: <work>/<contrast>/{counts.tsv.gz, coldata.tsv} and <work>/CONTRASTS.tsv.
Nothing is pooled across contrasts here; the label step combines them later.

Usage: [SPECIES='Glycine soja'] python3 -I de_prepare.py <work_dir> [stress ...]
"""
import gzip, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contrasts import ABIO, SHORT, SX, load_annotation, find_contrasts  # noqa: E402

WORK = sys.argv[1]
WANT = sys.argv[2:] or ["drought", "salt", "heat", "cold", "alkaline stress", "iron deficiency"]
SPECIES = os.environ.get("SPECIES", "Glycine max")      # "Glycine soja" for the wild-soybean (W05) cross-genome test
os.makedirs(WORK, exist_ok=True)

genes = [l.strip() for l in open(f"{SX}/data/expression/genes.txt") if l.strip()]
short = [g.split("ann1.")[-1] for g in genes]
loc = {r.run: (int(r.chunk), int(r.row)) for r in
       pd.read_csv(f"{SX}/data/expression/runs.tsv", sep="\t", dtype={"run": str}).itertuples()}
a = load_annotation()

contrasts = []
for s in WANT:
    contrasts += find_contrasts(a, s, min_per_arm=2, quantified_only=True, species=SPECIES)
need = sorted({r for c in contrasts for r in c["treated"] + c["control"]}, key=lambda r: loc[r])
print(f"{len(contrasts)} contrasts, {len(need)} distinct runs", flush=True)

# read every needed run once, chunk by chunk
counts = {}
cur, arr = None, None
for r in need:
    ch, row = loc[r]
    if ch != cur:
        arr = np.load(f"{SX}/data/expression/chunk_{ch:05d}.counts.npy", mmap_mode="r")
        cur = ch
    counts[r] = np.rint(np.asarray(arr[row], dtype=np.float64)).astype(np.int64)

man = []
for c in contrasts:
    d = f"{WORK}/{c['contrast']}"
    os.makedirs(d, exist_ok=True)
    runs = [(r, "control", t) for r, t in zip(c["control"], c["time_control"])] + \
           [(r, "treated", t) for r, t in zip(c["treated"], c["time_treated"])]
    M = np.stack([counts[r] for r, _, _ in runs], axis=1)
    with gzip.open(f"{d}/counts.tsv.gz", "wt", compresslevel=1) as o:
        o.write("gene\t" + "\t".join(r for r, _, _ in runs) + "\n")
        for j, g in enumerate(short):
            o.write(g + "\t" + "\t".join(map(str, M[j])) + "\n")
    with open(f"{d}/coldata.tsv", "w") as o:
        o.write("run\tcondition\ttime\n")
        for r, cond, t in runs:
            o.write(f"{r}\t{cond}\t{t or 'NA'}\n")
    man.append({k: c[k] for k in ("contrast", "stress", "study", "cultivar", "organ_group",
                                   "genetic_modification", "species", "tissue")}
               | {"n_treated": len(c["treated"]), "n_control": len(c["control"])})
pd.DataFrame(man).to_csv(f"{WORK}/CONTRASTS.tsv", sep="\t", index=False)
print(pd.DataFrame(man).groupby("stress").agg(contrasts=("contrast", "size"),
                                              studies=("study", "nunique")).to_string())
