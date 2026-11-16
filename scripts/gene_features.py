"""Gene-level covariates for the gene-feature baseline (plan step 5): no promoter sequence content.

Per Wm82.a4 gene:
  expression (JGI Gene Atlas runs in SoyExpFM, unstressed Williams 82 tissues): mean log2(TPM+1),
    max log2(TPM+1), breadth (share of runs with TPM > 1), tissue specificity tau;
  structure (primary transcript): transcript length, CDS length, exon count, 5'UTR and 3'UTR length,
    total intron length;
  window composition: GC share of the non-N bases in the TSS half and in the TTS half;
  homology context: protein family size (MMseqs2 30%), has a WGD partner, has a tandem partner.

Usage: python3 -I gene_features.py <project_dir>  -> features/gene_covariates.tsv
"""
import collections, gzip, sys
import numpy as np
import pandas as pd

P = sys.argv[1]
SX = "/home/user/zhangzhishuai/myhermes/2080/SoyExpFM"
genes_sx = [l.strip().split("ann1.")[-1] for l in open(f"{SX}/data/expression/genes.txt") if l.strip()]
runs = pd.read_csv(f"{SX}/data/expression/runs.tsv", sep="\t", dtype={"run": str})
ann = pd.read_csv(f"{SX}/metadata/sample_annotation.tsv", sep="\t", dtype=str).fillna("")
atlas = set(ann.run[ann.group == "atlas_jgi"])
r = runs[runs.run.isin(atlas)]
print(f"atlas runs quantified: {len(r)} of {len(atlas)}", flush=True)
rows = []
for ch, g in r.groupby("chunk"):
    tpm = np.load(f"{SX}/data/expression/chunk_{int(ch):05d}.tpm.npy", mmap_mode="r")
    rows.append(np.asarray(tpm[g.row.astype(int).values], dtype=np.float32))
T = np.vstack(rows)                                   # runs x genes
L = np.log2(T + 1)
mean_l, max_l = L.mean(0), L.max(0)
breadth = (T > 1).mean(0)
mx = np.maximum(L.max(0), 1e-9)
tau = ((1 - L / mx).sum(0) / max(L.shape[0] - 1, 1))
tau[L.max(0) == 0] = np.nan
expr = pd.DataFrame({"gene": genes_sx, "expr_mean_log2tpm": mean_l, "expr_max_log2tpm": max_l,
                     "expr_breadth": breadth, "expr_tau": tau})

# structure from the a4 GFF (primary transcript = longest=1)
gff = "/home/user/zhangzhishuai/myhermes/2080/SoyExpFM/ref/a4/genes.gff3"
mrna_gene, prim, feats = {}, {}, collections.defaultdict(lambda: collections.defaultdict(list))
for line in open(gff):
    if line.startswith("#"):
        continue
    f = line.rstrip("\n").split("\t")
    if len(f) < 9:
        continue
    a = dict(kv.split("=", 1) for kv in f[8].split(";") if "=" in kv)
    if f[2] == "mRNA":
        g = a["Parent"].split("ann1.")[-1]
        mrna_gene[a["ID"]] = g
        if a.get("longest") == "1":
            prim[g] = a["ID"]
    elif f[2] in ("CDS", "five_prime_UTR", "three_prime_UTR"):
        feats[a["Parent"]][f[2]].append((int(f[3]), int(f[4])))
st = []
for g, m in prim.items():
    d = feats[m]
    cds = sorted(d["CDS"]); u5 = d["five_prime_UTR"]; u3 = d["three_prime_UTR"]
    ex = sorted(cds + u5 + u3)
    merged = []
    for s, e in ex:                                   # UTR and CDS pieces of one exon touch: merge
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    tlen = sum(e - s + 1 for s, e in merged)
    span = merged[-1][1] - merged[0][0] + 1 if merged else 0
    st.append((g, tlen, sum(e - s + 1 for s, e in cds), len(merged),
               sum(e - s + 1 for s, e in u5), sum(e - s + 1 for s, e in u3), span - tlen))
struct = pd.DataFrame(st, columns=["gene", "tx_len", "cds_len", "n_exons", "utr5_len", "utr3_len", "intron_len"])

# window GC
gc = []
name = None
for line in gzip.open(f"{P}/seqs/wm82a4_windows.fa.gz", "rt"):
    if line.startswith(">"):
        name = line[1:].strip()
    else:
        s = line.strip()
        a, b = s[:1500], s[-1500:]
        def g_c(x):
            n = len(x) - x.count("N")
            return (x.count("G") + x.count("C")) / n if n else np.nan
        gc.append((name, g_c(a), g_c(b)))
gcd = pd.DataFrame(gc, columns=["gene", "gc_tss_half", "gc_tts_half"])

strip = lambda x: x.split("ann1.")[-1].rsplit(".", 1)[0]
fam = pd.read_csv(f"{P}/work/fam/a4_c30_cluster.tsv", sep="\t", header=None, names=["rep", "mem"])
fam["gene"] = fam.mem.map(strip); fam["rep"] = fam.rep.map(strip)
fam["family_size"] = fam.rep.map(fam.rep.value_counts())
wgd = pd.read_csv(f"{P}/results/wgd_pairs.tsv", sep="\t")
tan = pd.read_csv(f"{P}/results/tandem_pairs.tsv", sep="\t")
hom = pd.DataFrame({"gene": fam.gene, "family_size": fam.family_size})
hom["has_wgd_partner"] = hom.gene.isin(set(wgd.gene_a) | set(wgd.gene_b)).astype(int)
hom["has_tandem_partner"] = hom.gene.isin(set(tan.gene_a) | set(tan.gene_b)).astype(int)

out = gcd.merge(expr, on="gene", how="left").merge(struct, on="gene", how="left").merge(hom, on="gene", how="left")
out.to_csv(f"{P}/features/gene_covariates.tsv", sep="\t", index=False)
print(out.shape, "->", f"{P}/features/gene_covariates.tsv")
print(out.describe().T[["mean", "50%", "min", "max"]].round(3).to_string())
