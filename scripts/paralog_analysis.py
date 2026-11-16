"""WGD paralog pairs (plan step 7): how much of the response divergence between the two copies is predicted.

Pairs: results/wgd_pairs.tsv (reciprocal best hits in collinear blocks), without pairs whose proteins are >= 99%
identical (their reads are hard to assign, so their fold changes are unreliable). A pair is always in one fold, so
all predictions are test-set predictions.

Per task (stress x direction), on the pairs whose two copies both have a label for the task:
  (a) Spearman correlation between the predicted difference (copy A - copy B) and the measured difference in the
      meta-analysis log2 fold change (sign flipped for the down tasks); 95% CI from 1,000 bootstrap resamples of pairs;
  (b) divergent pairs (one copy responds, the other is non-responsive): paired AUROC = share of pairs in which the
      responding copy gets the higher prediction (ties count one half); bootstrap CI over pairs.
Models: StressCRE with the two copies' gene features replaced by their mean (the difference comes from the sequence
windows only; results/v1_pairs_sharedfeat.tsv.gz), StressCRE with each copy's own features, the k-mer model, the
gene-feature model and their stack (test-set predictions averaged over the three split seeds).
Pooled over the ten tasks: (a) mean of the per-task correlations and (b) AUROC over all divergent pairs, with
bootstrap CIs and paired bootstrap differences against the k-mer model.

Usage: python -I paralog_analysis.py <project>
Writes results/paralog_metrics.tsv, results/paralog_pooled.tsv and results/paralog_pairs.tsv.gz.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_models import TASKS   # noqa: E402
from cre_data import strict_calls   # noqa: E402

P = sys.argv[1]
B = 1000
rng = np.random.default_rng(0)
pairs = pd.read_csv(f"{P}/results/wgd_pairs.tsv", sep="\t")
pairs = pairs[pairs.protein_identity < 99].reset_index(drop=True)
lab = strict_calls(pd.read_csv(f"{P}/labels/labels_v1.tsv.gz", sep="\t"))
call = lab.set_index(["gene", "stress"]).call
lfc = lab.set_index(["gene", "stress"]).meta_lfc

# predictions: test-set probability per gene x task, mean over the three seeds
pred = {}
for m in ("kmer", "cov", "stack"):
    d = pd.concat([pd.read_csv(f"{P}/results/v1_{m}_main_seed{s}_predictions.tsv.gz", sep="\t") for s in range(3)])
    pred[m] = d.groupby(["gene", "task"]).pred.mean()
own = pd.read_csv(f"{P}/results/v1_stresscre_allgenes.tsv.gz", sep="\t").set_index(["gene", "task"]).p
pred["stresscre_own"] = own
sh = pd.read_csv(f"{P}/results/v1_pairs_sharedfeat.tsv.gz", sep="\t")
sh = sh.merge(pairs[["gene_a", "gene_b"]], on=["gene_a", "gene_b"])
MODELS = ["stresscre_shared", "stresscre_own", "kmer", "cov", "stack"]

rows, per_pair = [], []
for t in TASKS:
    s, d = t.split("-")
    q = pairs[["gene_a", "gene_b"]].copy()
    ca = [call.get((g, s), "none") for g in q.gene_a]
    cb = [call.get((g, s), "none") for g in q.gene_b]
    q["call_a"], q["call_b"] = ca, cb
    ok = q.call_a.isin([d, "nonresp"]) & q.call_b.isin([d, "nonresp"])
    q = q[ok].copy()
    sign = 1.0 if d == "up" else -1.0
    q["meas"] = sign * (np.array([lfc.get((g, s), np.nan) for g in q.gene_a]) - np.array([lfc.get((g, s), np.nan) for g in q.gene_b]))
    st = sh[sh.task == t].set_index(["gene_a", "gene_b"])
    q["stresscre_shared"] = (st.pa - st.pb).reindex(pd.MultiIndex.from_frame(q[["gene_a", "gene_b"]])).values
    for m in MODELS[1:]:
        pa = pred[m].reindex(pd.MultiIndex.from_arrays([q.gene_a, [t] * len(q)])).values
        pb = pred[m].reindex(pd.MultiIndex.from_arrays([q.gene_b, [t] * len(q)])).values
        q[m] = pa - pb
    q = q.dropna(subset=["meas"] + MODELS)
    q["task"] = t
    q["divergent"] = ((q.call_a == d) & (q.call_b == "nonresp")) | ((q.call_a == "nonresp") & (q.call_b == d))
    # orientation for the paired AUROC: + if copy a is the responding one
    q["resp_sign"] = np.where(q.call_a == d, 1.0, -1.0)
    per_pair.append(q)
    dv = q[q.divergent]
    idx_all = np.arange(len(q))
    for m in MODELS:
        r = spearmanr(q[m], q.meas).correlation
        bs = []
        for _ in range(B):
            ii = rng.choice(idx_all, len(idx_all))
            bs.append(spearmanr(q[m].values[ii], q.meas.values[ii]).correlation)
        x = dv[m].values * dv.resp_sign.values
        auc = float(np.mean((x > 0) + 0.5 * (x == 0))) if len(dv) else np.nan
        ba = []
        for _ in range(B if len(dv) else 0):
            ii = rng.integers(0, len(dv), len(dv))
            xx = x[ii]
            ba.append(np.mean((xx > 0) + 0.5 * (xx == 0)))
        rows.append(dict(task=t, model=m, n_pairs=len(q), spearman=round(r, 4),
                         spearman_lo=round(np.nanpercentile(bs, 2.5), 4), spearman_hi=round(np.nanpercentile(bs, 97.5), 4),
                         n_divergent=len(dv), pair_auroc=round(auc, 4),
                         auroc_lo=round(np.percentile(ba, 2.5), 4) if ba else np.nan,
                         auroc_hi=round(np.percentile(ba, 97.5), 4) if ba else np.nan))
    print(t, len(q), "pairs,", len(dv), "divergent", flush=True)
met = pd.DataFrame(rows)
met.to_csv(f"{P}/results/paralog_metrics.tsv", sep="\t", index=False)
pp = pd.concat(per_pair)
pp.to_csv(f"{P}/results/paralog_pairs.tsv.gz", sep="\t", index=False)

# pooled: mean per-task Spearman and AUROC over all divergent pairs; paired bootstrap vs k-mer (resampling pairs within tasks)
groups = {t: g.reset_index(drop=True) for t, g in pp.groupby("task")}
dvall = pp[pp.divergent].reset_index(drop=True)


def pooled(m, ii_by_task=None, ii_div=None):
    rs = []
    for t, g in groups.items():
        ii = ii_by_task[t] if ii_by_task else np.arange(len(g))
        rs.append(spearmanr(g[m].values[ii], g.meas.values[ii]).correlation)
    jj = ii_div if ii_div is not None else np.arange(len(dvall))
    x = dvall[m].values[jj] * dvall.resp_sign.values[jj]
    return float(np.nanmean(rs)), float(np.mean((x > 0) + 0.5 * (x == 0)))


base = {m: pooled(m) for m in MODELS}
boot = {m: [] for m in MODELS}
for _ in range(B):
    ii = {t: rng.integers(0, len(g), len(g)) for t, g in groups.items()}
    jj = rng.integers(0, len(dvall), len(dvall))
    for m in MODELS:
        boot[m].append(pooled(m, ii, jj))
out = []
for m in MODELS:
    b = np.array(boot[m]); k = np.array(boot["kmer"])
    out.append(dict(model=m, mean_spearman=round(base[m][0], 4), spearman_lo=round(np.percentile(b[:, 0], 2.5), 4),
                    spearman_hi=round(np.percentile(b[:, 0], 97.5), 4), n_divergent=len(dvall),
                    pair_auroc=round(base[m][1], 4), auroc_lo=round(np.percentile(b[:, 1], 2.5), 4), auroc_hi=round(np.percentile(b[:, 1], 97.5), 4),
                    diff_spearman_vs_kmer=round(base[m][0] - base["kmer"][0], 4),
                    diff_sp_lo=round(np.percentile(b[:, 0] - k[:, 0], 2.5), 4), diff_sp_hi=round(np.percentile(b[:, 0] - k[:, 0], 97.5), 4),
                    diff_auroc_vs_kmer=round(base[m][1] - base["kmer"][1], 4),
                    diff_auc_lo=round(np.percentile(b[:, 1] - k[:, 1], 2.5), 4), diff_auc_hi=round(np.percentile(b[:, 1] - k[:, 1], 97.5), 4)))
pooled_df = pd.DataFrame(out)
pooled_df.to_csv(f"{P}/results/paralog_pooled.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(pooled_df.to_string(index=False))
print(met[met.model.isin(["stresscre_shared", "kmer"])].pivot(index="task", columns="model", values=["spearman", "pair_auroc"]).to_string())
