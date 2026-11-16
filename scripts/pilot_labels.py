"""Pilot labels: how many genes come out up / down / non-responsive per stress (2026-11-10).

Inputs: <work>/CONTRASTS.tsv and <work>/<contrast>/de.tsv.gz from de_run.R.
Contrasts are first collapsed to studies (a study calls a gene up if any of its contrasts has
padj < 0.05 and log2FC > 1 and none calls it down; its log2FC is the inverse-variance mean).
Two gene-level rules are compared:
  vote : up   = up in >= 2 studies and in >= 1/3 of the studies that tested it, down in none
         (down symmetric);
         non-responsive = tested in >= 3 studies, never padj < 0.05 with |log2FC| > 0.5,
                          and |median study log2FC| < 0.3.
  meta : DerSimonian-Laird random-effects mean of the study log2FCs (genes tested in >= 2
         studies), BH over genes; up = q < 0.05 and mean > 0.585 (1.5-fold); down symmetric;
         non-responsive = q > 0.5, |mean| < 0.3, tested in >= 3 studies.
Strict labels (used downstream) = both rules agree for up/down; non-responsive = meta rule and the
vote rule calls neither up nor down.
Writes results/pilot_label_counts.tsv, results/pilot_label_overlap.tsv, labels/pilot_labels.tsv.gz.

Usage: python3 -I pilot_labels.py <work_dir> <project_dir> [tag=pilot]
"""
import os, sys
from math import sqrt
import numpy as np
import pandas as pd
from scipy.special import erfc

W, P = sys.argv[1:3]
TAG = sys.argv[3] if len(sys.argv) > 3 else "pilot"   # output names: labels/<TAG>_labels.tsv.gz etc.
SX = "/home/user/zhangzhishuai/myhermes/2080/SoyExpFM"
man = pd.read_csv(f"{W}/CONTRASTS.tsv", sep="\t", dtype=str)
man = man[man.species == "Glycine max"]          # wild-soybean studies are kept for testing
genes = [l.strip().split("ann1.")[-1] for l in open(f"{SX}/data/expression/genes.txt") if l.strip()]
gi = {g: i for i, g in enumerate(genes)}
G = len(genes)
UP, LFC_SIG, MID = 1.0, 0.5, 0.3


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


rows, frames, sets = [], [], {}
for stress, cm in man.groupby("stress"):
    studies = sorted(cm.study.unique())
    S = len(studies)
    sw = np.zeros((S, G)); swl = np.zeros((S, G))           # sum of weights, sum of w*lfc
    up = np.zeros((S, G), bool); dn = np.zeros((S, G), bool); sig = np.zeros((S, G), bool)
    for c in cm.itertuples():
        k = studies.index(c.study)
        d = pd.read_csv(f"{W}/{c.contrast}/de.tsv.gz", sep="\t").dropna(subset=["log2FoldChange", "lfcSE"])
        d = d[(d.lfcSE > 0) & d.gene.isin(gi)]
        ix = d.gene.map(gi).values
        lfc = d.log2FoldChange.values; w = 1 / d.lfcSE.values ** 2
        padj = d.padj.fillna(1).values
        sw[k, ix] += w; swl[k, ix] += w * lfc
        up[k, ix] |= (padj < 0.05) & (lfc > UP)
        dn[k, ix] |= (padj < 0.05) & (lfc < -UP)
        sig[k, ix] |= (padj < 0.05) & (np.abs(lfc) > LFC_SIG)
    tested = sw > 0
    lfc_s = np.where(tested, swl / np.where(tested, sw, 1), np.nan)
    var_s = np.where(tested, 1 / np.where(tested, sw, 1), np.nan)
    upc, dnc = up & ~dn, dn & ~up
    n_t = tested.sum(0); n_up = upc.sum(0); n_dn = dnc.sum(0); n_sig = sig.sum(0)
    med = np.nanmedian(np.where(tested, lfc_s, np.nan), axis=0) if S > 0 else np.full(G, np.nan)
    vote = np.full(G, "unclear", dtype=object)
    vote[(n_up >= 2) & (n_up >= n_t / 3) & (n_dn == 0)] = "up"
    vote[(n_dn >= 2) & (n_dn >= n_t / 3) & (n_up == 0)] = "down"
    vote[(n_t >= 3) & (n_sig == 0) & (np.abs(med) < MID)] = "nonresp"
    vote[n_t == 0] = "untested"
    # vectorised DerSimonian-Laird over studies
    w = np.where(tested, 1 / np.where(tested, var_s, 1), 0.0)
    y = np.where(tested, lfc_s, 0.0)
    sw1 = w.sum(0); sw2 = (w ** 2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_f = (w * y).sum(0) / sw1
        Q = (w * (y - mu_f) ** 2).sum(0)
        C = sw1 - sw2 / sw1
        tau2 = np.maximum(0, (Q - (n_t - 1)) / C)
        ws = np.where(tested, 1 / (np.where(tested, var_s, 1) + tau2), 0.0)
        mu = (ws * y).sum(0) / ws.sum(0)
        se = np.sqrt(1 / ws.sum(0))
    ok = n_t >= 2
    p = np.ones(G); p[ok] = erfc(np.abs(mu[ok] / se[ok]) / sqrt(2))
    q = np.ones(G); q[ok] = bh(p[ok])
    meta = np.full(G, "unclear", dtype=object)
    meta[ok & (q < 0.05) & (mu > 0.585)] = "up"
    meta[ok & (q < 0.05) & (mu < -0.585)] = "down"
    meta[ok & (q > 0.5) & (np.abs(mu) < MID) & (n_t >= 3)] = "nonresp"
    meta[n_t == 0] = "untested"
    lab = pd.DataFrame({"gene": genes, "stress": stress, "n_tested": n_t, "n_up": n_up, "n_dn": n_dn,
                        "n_sig": n_sig, "med_lfc": np.round(med, 3), "meta_lfc": np.round(mu, 3),
                        "meta_q": q, "vote": vote, "meta": meta})
    frames.append(lab[lab.n_tested > 0])
    vc, mc = pd.Series(vote).value_counts(), pd.Series(meta).value_counts()
    strict_nr = int(((meta == "nonresp") & ~np.isin(vote, ["up", "down"])).sum())
    rows.append((stress, S, len(cm), int((n_t > 0).sum()), int((n_t >= 3).sum()),
                 int(vc.get("up", 0)), int(vc.get("down", 0)), int(vc.get("nonresp", 0)),
                 int(mc.get("up", 0)), int(mc.get("down", 0)), int(mc.get("nonresp", 0)),
                 int(((vote == "up") & (meta == "up")).sum()), int(((vote == "down") & (meta == "down")).sum()),
                 strict_nr))
    sets[stress] = {"up": set(np.array(genes)[vote == "up"]), "down": set(np.array(genes)[vote == "down"]),
                    "strict_up": set(np.array(genes)[(vote == "up") & (meta == "up")]),
                    "strict_down": set(np.array(genes)[(vote == "down") & (meta == "down")])}
    print(stress, "done", flush=True)

cnt = pd.DataFrame(rows, columns=["stress", "studies", "contrasts", "genes_tested", "genes_tested_3plus",
                                  "vote_up", "vote_down", "vote_nonresp", "meta_up", "meta_down",
                                  "meta_nonresp", "strict_up", "strict_down", "strict_nonresp"])
os.makedirs(f"{P}/results", exist_ok=True); os.makedirs(f"{P}/labels", exist_ok=True)
cnt.to_csv(f"{P}/results/{TAG}_label_counts.tsv", sep="\t", index=False)
pd.concat(frames).to_csv(f"{P}/labels/{TAG}_labels.tsv.gz", sep="\t", index=False)
names = sorted(sets)
ov = [[d, i] + [len(sets[i][d] & sets[j][d]) for j in names] for d in ("up", "down", "strict_up", "strict_down") for i in names]
ov = pd.DataFrame(ov, columns=["direction", "stress"] + names)
ov.to_csv(f"{P}/results/{TAG}_label_overlap.tsv", sep="\t", index=False)
allup = set().union(*[sets[s]["up"] for s in names]); alldn = set().union(*[sets[s]["down"] for s in names])
pd.set_option("display.width", 250)
print(cnt.to_string(index=False))
print(ov.to_string(index=False))
print(f"genes up in >= 1 stress: {len(allup)}; down in >= 1: {len(alldn)}; "
      f"up in one stress and down in another: {len(allup & alldn)}")
