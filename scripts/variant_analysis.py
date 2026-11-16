"""Population patterns of the predicted promoter/terminator variant effects (plan step 8).

Inputs: variants/panel_snps.tsv.gz (SNP x gene rows), the deltas variants/delta_<cnn|pc2>[.shardN].npy (change in
predicted probability, alternative - reference allele, ten tasks, mean over the three test-fold models), the allele
counts in wild (G. soja, 203) and cultivated (G. max, 1,308) accessions (variants/af_soja|max.tsv.gz) and StressCRE's
predictions for all genes (results/v1_stresscre_allgenes.tsv.gz, gene baseline).
Two sets of rows:
  StressCRE  : rows scored by both networks (the 300,000-row subset), delta = mean of the two networks;
  CNN, all   : all rows, convolutional network alone (robustness check).
Effect of a SNP on a gene = largest |delta| over the nine main tasks; large effect = top 1% of the rows of the set.
Strata for Cochran-Mantel-Haenszel odds ratios: window position (100-bp bins) x quintile of the gene's baseline
(largest StressCRE prediction over the nine main tasks) x CpG (SNP in a CpG of either allele; CpG sites mutate
faster), so that neither the distance to the TSS/TTS, the gene's predicted probability nor CpG mutability drives
the result.
  (1) frequency: odds ratio of being rare (MAF < 5%) for large-effect vs other SNPs, over all accessions and within
      the wild and the cultivated group (SNPs polymorphic in that group);
  (2) domestication: odds ratio of large effect for differentiated (|AF cultivated - AF wild| >= 0.5) vs other
      common SNPs (MAF >= 5% over all accessions);
  (3) elements: SNPs that create or destroy an ABRE (ACGTGG/T, either strand, overlapping the SNP; a2 window
      sequence): odds ratio for large-effect vs other SNPs, and the share of such SNPs whose effect on the five
      up-regulation tasks (mean delta) has the expected sign (destroyed -> lower, created -> higher), and the odds
      ratio of being rare for ABRE-destroying (or -creating) vs other SNPs (n_large column = SNPs of that kind);
  (4) per gene (StressCRE set): number of common large-effect SNPs.
Usage: python -I variant_analysis.py <project>
Writes results/variant_summary.tsv, results/variant_genes.tsv.gz, variants/scored_snps.tsv.gz (StressCRE set).
"""
import glob, gzip, os, re, sys
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import StratifiedTable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_models import TASKS   # noqa: E402

P = sys.argv[1]
MAIN = [j for j, t in enumerate(TASKS) if t != "alkaline-down"]
snp0 = pd.read_csv(f"{P}/variants/panel_snps.tsv.gz", sep="\t")


def load_delta(comp):
    f = f"{P}/variants/delta_{comp}.npy"
    if os.path.exists(f):
        return np.load(f).astype(np.float32)
    parts = sorted(glob.glob(f"{P}/variants/delta_{comp}.shard*.npy"))
    if not parts:
        return None
    d = np.full((len(snp0), len(TASKS)), np.nan, np.float32)
    for p in parts:
        x = np.load(p).astype(np.float32)
        ok = ~np.isnan(x[:, 0])
        d[ok] = x[ok]
    return d


af = {}
for g in ("soja", "max"):
    a = pd.read_csv(f"{P}/variants/af_{g}.tsv.gz", sep="\t", header=None, names=["chrom", "pos", "ref", "alt", "ac", "an"])
    a = a[~a.alt.str.contains(",")]
    a["ac"] = pd.to_numeric(a.ac, errors="coerce")
    a["an"] = pd.to_numeric(a.an, errors="coerce")
    af[g] = a.drop_duplicates(["chrom", "pos"]).set_index(["chrom", "pos"])
key = pd.MultiIndex.from_arrays([snp0.chrom, snp0.pos])
for g in ("soja", "max"):
    snp0[f"ac_{g}"] = af[g].ac.reindex(key).values
    snp0[f"an_{g}"] = af[g].an.reindex(key).values
snp0["af_soja"] = snp0.ac_soja / snp0.an_soja
snp0["af_max"] = snp0.ac_max / snp0.an_max
snp0["af_all"] = (snp0.ac_soja + snp0.ac_max) / (snp0.an_soja + snp0.an_max)
snp0["maf"] = np.minimum(snp0.af_all, 1 - snp0.af_all)
snp0["maf_soja"] = np.minimum(snp0.af_soja, 1 - snp0.af_soja)
snp0["maf_max"] = np.minimum(snp0.af_max, 1 - snp0.af_max)
base = pd.read_csv(f"{P}/results/v1_stresscre_allgenes.tsv.gz", sep="\t").pivot(index="gene", columns="task", values="p")
base = base[[TASKS[j] for j in MAIN]].max(1)
snp0["baseline"] = base.reindex(snp0.gene_a4).values
usable = (snp0.an_soja > 0) & (snp0.an_max > 0) & (snp0.maf > 0) & snp0.baseline.notna()   # polymorphic in the panel
COMP = str.maketrans("ACGTN", "TGCAN")
ABRE = re.compile("(?=(ACGTG[GT]|[AC]CACGT))")


def abre_at(seq, c):
    return any(m.start() <= c < m.start() + 6 for m in ABRE.finditer(seq))


win, name = {}, None
with gzip.open(f"{P}/seqs/wm82a2_windows.fa.gz", "rt") as f:
    for line in f:
        if line.startswith(">"):
            name = line[1:].strip()
            win[name] = []
        else:
            win[name].append(line.strip())
win = {k: "".join(v) for k, v in win.items()}
cls, cpg = [], []
for g, i, a, st in zip(snp0.gene_a2, snp0.idx, snp0.alt, snp0.strand):
    w = win.get(g)
    lo = max(0, i - 5)
    loc = w[lo:i + 6] if w else ""
    c = i - lo
    if len(loc) <= c:
        cls.append("none")
        cpg.append(False)
        continue
    alt = a if st == "+" else a.translate(COMP)
    loc_alt = loc[:c] + alt + loc[c + 1:]
    r_, a_ = abre_at(loc, c), abre_at(loc_alt, c)
    cls.append("destroyed" if r_ and not a_ else ("created" if a_ and not r_ else "none"))
    cpg.append("CG" in loc[max(0, c - 1):c + 2] or "CG" in loc_alt[max(0, c - 1):c + 2])   # SNP in a CpG of either allele
snp0["abre"] = cls
snp0["cpg"] = cpg
del win
UPJ = [j for j, t in enumerate(TASKS) if t.endswith("-up")]
print("ABRE changes:", snp0.abre.value_counts().to_dict(), flush=True)
print(f"rows {len(snp0)}; usable (allele counts in both groups, polymorphic, gene baseline): {int(usable.sum())}", flush=True)


def cmh(snp, mask_a, mask_b, sub):
    """CMH odds ratio of mask_b given mask_a over the strata, within rows `sub`."""
    tabs = []
    for _, g in snp[sub].groupby("stratum"):
        a, b = mask_a[g.index], mask_b[g.index]
        t = np.array([[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]], float)
        if t.sum(1).min() > 0 and t.sum(0).min() > 0:
            tabs.append(t)                  # Mantel-Haenszel handles zero cells; no continuity correction
    st = StratifiedTable(tabs)
    lo, hi = st.oddsratio_pooled_confint()
    return st.oddsratio_pooled, lo, hi, st.test_null_odds().pvalue, len(tabs)


def analyse(name, delta):
    ok = usable.values & ~np.isnan(delta[:, 0])
    snp = snp0[ok].reset_index(drop=True)
    d = delta[ok]
    snp["effect"] = np.abs(d[:, MAIN]).max(1)
    snp["task_max"] = np.array([TASKS[MAIN[j]] for j in np.abs(d[:, MAIN]).argmax(1)])
    for j, t in enumerate(TASKS):
        snp[f"d_{t}"] = d[:, j].round(6)
    snp["stratum"] = ((snp.idx // 100).astype(str) + "_" + pd.qcut(snp.baseline, 5, labels=False, duplicates="drop").astype(str)
                      + "_" + snp.cpg.astype(int).astype(str))
    thr = snp.effect.quantile(0.99)
    snp["large"] = snp.effect >= thr
    large = snp.large.values
    rows = []
    for grp, col in (("all accessions", "maf"), ("wild", "maf_soja"), ("cultivated", "maf_max")):
        poly = snp[col].values > 0
        rare = snp[col].values < 0.05
        o, lo, hi, p, k = cmh(snp, large, rare, poly)
        rows.append(dict(model=name, analysis=f"large effect -> rare (MAF<5%), {grp}", n=int(poly.sum()), n_large=int((large & poly).sum()),
                         share_a=round(rare[large & poly].mean(), 4), share_b=round(rare[~large & poly].mean(), 4),
                         odds_ratio=round(o, 5), ci_lo=round(lo, 5), ci_hi=round(hi, 5), p=float(f"{p:.3g}"), strata=k, threshold=round(thr, 5)))
    noab = (snp.abre == "none").values
    for grp, col in (("all accessions", "maf"), ("wild", "maf_soja"), ("cultivated", "maf_max")):
        poly = (snp[col].values > 0) & noab
        rare = snp[col].values < 0.05
        o, lo, hi, p, k = cmh(snp, large, rare, poly)
        rows.append(dict(model=name, analysis=f"large effect without ABRE change -> rare (MAF<5%), {grp}", n=int(poly.sum()),
                         n_large=int((large & poly).sum()), share_a=round(rare[large & poly].mean(), 4), share_b=round(rare[~large & poly].mean(), 4),
                         odds_ratio=round(o, 5), ci_lo=round(lo, 5), ci_hi=round(hi, 5), p=float(f"{p:.3g}"), strata=k, threshold=round(thr, 5)))
    common = snp.maf.values >= 0.05
    diff = np.abs(snp.af_max.values - snp.af_soja.values) >= 0.5
    o, lo, hi, p, k = cmh(snp, diff, large, common & noab)
    rows.append(dict(model=name, analysis="differentiated -> large effect without ABRE change, common SNPs", n=int((common & noab).sum()),
                     n_large=int((large & common & noab).sum()), share_a=round(large[diff & common & noab].mean(), 4),
                     share_b=round(large[~diff & common & noab].mean(), 4), odds_ratio=round(o, 5), ci_lo=round(lo, 5),
                     ci_hi=round(hi, 5), p=float(f"{p:.3g}"), strata=k, threshold=round(thr, 5)))
    o, lo, hi, p, k = cmh(snp, diff, large, common)
    rows.append(dict(model=name, analysis="differentiated (|AF cult - AF wild| >= 0.5) -> large effect, common SNPs", n=int(common.sum()),
                     n_large=int((large & common).sum()), share_a=round(large[diff & common].mean(), 4),
                     share_b=round(large[~diff & common].mean(), 4), odds_ratio=round(o, 5), ci_lo=round(lo, 5),
                     ci_hi=round(hi, 5), p=float(f"{p:.3g}"), strata=k, threshold=round(thr, 5)))
    ab = (snp.abre != "none").values
    allr = np.ones(len(snp), bool)
    o_, lo_, hi_, p_, k_ = cmh(snp, large, ab, allr)
    rows.append(dict(model=name, analysis="large effect -> creates or destroys an ABRE", n=len(snp), n_large=int(large.sum()),
                     share_a=round(ab[large].mean(), 4), share_b=round(ab[~large].mean(), 4), odds_ratio=round(o_, 5),
                     ci_lo=round(lo_, 5), ci_hi=round(hi_, 5), p=float(f"{p_:.3g}"), strata=k_, threshold=round(thr, 5)))
    for grp, col in (("all accessions", "maf"), ("wild", "maf_soja"), ("cultivated", "maf_max")):
        poly = snp[col].values > 0
        rare = snp[col].values < 0.05
        for kind in ("destroyed", "created"):
            m = (snp.abre == kind).values
            o_, lo_, hi_, p_, k_ = cmh(snp, m, rare, poly)
            rows.append(dict(model=name, analysis=f"ABRE {kind} -> rare (MAF<5%), {grp}", n=int(poly.sum()), n_large=int((m & poly).sum()),
                             share_a=round(rare[m & poly].mean(), 4), share_b=round(rare[~m & poly].mean(), 4), odds_ratio=round(o_, 5),
                             ci_lo=round(lo_, 5), ci_hi=round(hi_, 5), p=float(f"{p_:.3g}"), strata=k_, threshold=round(thr, 5)))
    dup = d[:, UPJ].mean(1)
    for kind, sign in (("destroyed", -1), ("created", 1)):
        m = (snp.abre == kind).values
        for sub, lab in ((m, "all"), (m & large, "large")):
            if sub.sum():
                rows.append(dict(model=name, analysis=f"ABRE {kind} ({lab}): share with expected sign of mean up-task delta",
                                 n=int(sub.sum()), n_large=int((sub & large).sum()), share_a=round(float((np.sign(dup[sub]) == sign).mean()), 4),
                                 share_b=round(float((np.sign(dup[~m]) == sign).mean()), 4), odds_ratio=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                                 p=np.nan, strata=0, threshold=round(thr, 5)))
    print(f"{name}: {len(snp)} rows, threshold {thr:.5f}", flush=True)
    return snp, rows


dc, dp = load_delta("cnn"), load_delta("pc2")
out = []
if dp is not None:
    snp, r = analyse("StressCRE", (dc + dp) / 2)
    out += r
    common, large = snp.maf.values >= 0.05, snp.large.values
    gene = snp[common & large].groupby("gene_a4").agg(n_large_common=("pos", "size"), max_effect=("effect", "max"),
                                                       task=("task_max", lambda x: x.value_counts().index[0]))
    gene = gene.join(snp[common].groupby("gene_a4").size().rename("n_common_scored")).sort_values("n_large_common", ascending=False)
    gene.to_csv(f"{P}/results/variant_genes.tsv.gz", sep="\t")
    keep = ["chrom", "pos", "ref", "alt", "gene_a2", "gene_a4", "idx", "strand", "effect", "task_max", "af_soja", "af_max",
            "maf", "baseline", "abre", "cpg", "large"] + [f"d_{t}" for t in TASKS]
    snp[keep].to_csv(f"{P}/variants/scored_snps.tsv.gz", sep="\t", index=False)
    print(f"genes with >= 1 common large-effect SNP: {len(gene)}")
_, r = analyse("CNN, all rows", dc)
out += r
summ = pd.DataFrame(out)
summ.to_csv(f"{P}/results/variant_summary.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(summ.drop(columns=["threshold"]).to_string(index=False))
