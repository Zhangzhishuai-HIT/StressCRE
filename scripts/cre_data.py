"""Shared data loading for the steps after training (attribution, paralog pairs, variant scoring).
Reproduces exactly what train_cre.py does (checked: identical arrays), so trained models can be reused.

read_windows(fa)          -> names, seqs (3,020-bp windows, FASTA order)
labels_matrix(lab, names) -> Y [N, 10] in {1, 0, -1}; the strict up/down vs non-responsive rule
covariates(cov_file, names) -> C [N, 16] float32 (lengths log1p, tau-missing flag), before z-scoring
zscore(C, mu, sd)         -> covariates as the network saw them (mu/sd saved in models/<name>/fold<k>.pt)
load_fold_model(path, n_cov, device) -> (model, mu, sd, kind)
"""
import gzip
import numpy as np
import pandas as pd
import torch

from cre_models import CNN, PC2Head, TASKS

LOG_COLS = ["tx_len", "cds_len", "n_exons", "utr5_len", "utr3_len", "intron_len", "family_size"]


def read_windows(fa):
    names, seqs = [], []
    op = gzip.open if fa.endswith(".gz") else open
    for line in op(fa, "rt"):
        if line.startswith(">"):
            names.append(line[1:].strip())
        else:
            seqs.append(line.strip())
    return names, seqs


def strict_calls(lab):
    lab = lab.copy()
    lab["call"] = "unclear"
    lab.loc[(lab.vote == "up") & (lab.meta == "up"), "call"] = "up"
    lab.loc[(lab.vote == "down") & (lab.meta == "down"), "call"] = "down"
    lab.loc[(lab.meta == "nonresp") & ~lab.vote.isin(["up", "down"]), "call"] = "nonresp"
    return lab


def labels_matrix(lab, names):
    gi = {g: i for i, g in enumerate(names)}
    lab = strict_calls(lab[lab.stress.isin({t.split("-")[0] for t in TASKS})])
    Y = np.full((len(names), len(TASKS)), -1, np.int8)
    for j, t in enumerate(TASKS):
        s, d = t.split("-")
        q = lab[(lab.stress == s) & lab.call.isin([d, "nonresp"]) & lab.gene.isin(set(names))]
        Y[q.gene.map(gi).values, j] = (q.call == d).astype(np.int8).values
    return Y


def covariates(cov_file, names):
    cov = pd.read_csv(cov_file, sep="\t").set_index("gene").loc[names]
    C = cov.copy()
    for c in LOG_COLS:
        C[c] = np.log1p(C[c])
    C["tau_missing"] = cov.expr_tau.isna().astype(float)
    return C.values.astype(np.float32)


def zscore(C, mu, sd):
    return np.nan_to_num((C - mu) / sd).astype(np.float32)


def load_fold_model(path, n_cov, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    kind = ck["model"]
    drop = ck["hyper"]["drop"]
    m = CNN(n_cov, len(TASKS), drop) if kind == "cnn" else PC2Head(n_cov, len(TASKS), drop=drop)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck["cov_mu"], ck["cov_sd"], kind
