"""DeepCRE soybean models as a baseline (plan step 5, table 4): can a model of expression level predict the
stress response?

The NAMlab DeepCRE web tool's soybean leaf and root models (Keras 2.10 h5 files in
down_data/GitHub_NAMlab_deepCRE_soybean) are rebuilt in PyTorch from the h5 weights, without TensorFlow:
three blocks of two Conv1D(k=8, 'same') + ReLU and MaxPool1D(8, 'same') (dropout is inactive at inference),
Flatten (channels-last order), Dense 128 / 64 / 1, sigmoid. TensorFlow 'same' padding: convolutions pad 3 left
and 4 right; pooling pads floor(total/2) left.
Inputs follow the tool's own extraction (streamlit_deepcre_app lib/utils.py prepare_dataset, HEAD 2026-11-16):
one-hot A C G T (other = 0); promoter TSS-1000 .. TSS+ext, ext = min(500, gene length // 2); zero padding to
3,020 bp; terminator TTS-ext .. TTS+1000; transcript orientation; coding sequence NOT masked (as in DeepCRE).
TSS/TTS = ends of the primary transcript (seqs/wm82a4_windows.tsv).

Check of the rebuild: the tool's models were trained to separate high- from low-expressed genes, so their score
must rank the JGI atlas expression; the top- vs bottom-quartile AUROC is printed.

Baseline per task: logistic regression on the two scores' logits, fitted on the training folds and applied to
the test fold (main 5-fold and random split: per fold; chromosome hold-out: per round).

Usage: python -I deepcre_baseline.py score <project> <genome.fa.gz> <h5_dir>        -> features/deepcre_soy_scores.tsv
       python -I deepcre_baseline.py fit <project> <split> <labels.tsv.gz> <out_prefix> -> <out_prefix>_predictions.tsv.gz, _metrics.tsv
"""
import gzip, os, sys, time
import numpy as np
import pandas as pd

MODE, P = sys.argv[1:3]
STRESSES = ["drought", "salt", "heat", "cold", "alkaline"]


def score(fa, h5dir):
    import h5py
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))

    class DeepCRE(nn.Module):
        def __init__(self, h5):
            super().__init__()
            f = h5py.File(h5, "r")["model_weights"]
            w = lambda n, k: np.asarray(f[n][n][k + ":0"])
            self.convs = nn.ModuleList()
            for i in range(6):
                n = "conv1d" if i == 0 else f"conv1d_{i}"
                k = w(n, "kernel")                                    # (8, in, out)
                c = nn.Conv1d(k.shape[1], k.shape[2], k.shape[0])
                c.weight.data = torch.from_numpy(np.ascontiguousarray(k.transpose(2, 1, 0)))
                c.bias.data = torch.from_numpy(w(n, "bias"))
                self.convs.append(c)
            self.dense = nn.ModuleList()
            for i in range(3):
                n = "dense" if i == 0 else f"dense_{i}"
                k = w(n, "kernel")                                    # (in, out)
                d = nn.Linear(k.shape[0], k.shape[1])
                d.weight.data = torch.from_numpy(np.ascontiguousarray(k.T))
                d.bias.data = torch.from_numpy(w(n, "bias"))
                self.dense.append(d)

        def forward(self, x):                                         # x [B, 4, 3020]
            for b in range(3):
                for j in range(2):
                    x = F.relu(self.convs[2 * b + j](F.pad(x, (3, 4))))
                L = x.shape[-1]
                tot = max((-(-L // 8) - 1) * 8 + 8 - L, 0)
                x = F.max_pool1d(F.pad(x, (tot // 2, tot - tot // 2), value=float("-inf")), 8)
            x = x.permute(0, 2, 1).reshape(x.shape[0], -1)            # channels-last flatten (Keras)
            x = F.relu(self.dense[0](x))
            x = F.relu(self.dense[1](x))
            return torch.sigmoid(self.dense[2](x)).squeeze(1)

    t0 = time.time()
    genome, name, buf = {}, None, []
    for line in gzip.open(fa, "rt"):
        if line.startswith(">"):
            if name:
                genome[name] = "".join(buf).upper()
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    genome[name] = "".join(buf).upper()
    print(f"genome: {len(genome)} sequences in {time.time() - t0:.0f}s", flush=True)
    comp = str.maketrans("ACGTN", "TGCAN")
    rc = lambda s: s.translate(comp)[::-1]

    def grab(ch, a, b):                                               # 0-based half-open, N beyond the ends
        s = genome[ch]
        return "N" * max(0, -a) + s[max(a, 0):min(b, len(s))] + "N" * max(0, b - len(s))

    win = pd.read_csv(f"{P}/seqs/wm82a4_windows.tsv", sep="\t")
    seqs = []
    for r in win.itertuples():
        lo, hi = (r.tss, r.tts) if r.strand == "+" else (r.tts, r.tss)  # 1-based inclusive gene span
        s0, e0 = lo - 1, hi
        ext = min(500, (e0 - s0) // 2)
        prom, term = grab(r.chrom, s0 - 1000, s0 + ext), grab(r.chrom, e0 - ext, e0 + 1000)
        pad = "N" * (3020 - len(prom) - len(term))
        seqs.append(prom + pad + term if r.strand == "+" else rc(term) + pad + rc(prom))
    assert all(len(s) == 3020 for s in seqs)
    lut = np.full(256, 4, np.uint8)
    for i, b in enumerate("ACGT"):
        lut[ord(b)] = i
    codes = np.stack([lut[np.frombuffer(s.encode(), np.uint8)] for s in seqs])
    del genome
    out = {}
    for tissue in ("leaf", "root"):
        m = DeepCRE(os.path.join(h5dir, f"Glycine_max_{tissue}_v26J.h5")).eval()
        p = []
        with torch.no_grad():
            for s in range(0, len(codes), 256):
                x = F.one_hot(torch.from_numpy(codes[s:s + 256]).long(), 5)[..., :4].permute(0, 2, 1).float()
                p.append(m(x).numpy())
        out[tissue] = np.concatenate(p)
        print(f"{tissue}: scored {len(out[tissue])} genes  [{time.time() - t0:.0f}s]", flush=True)
    res = pd.DataFrame({"gene": win.gene, "deepcre_leaf": out["leaf"].round(6), "deepcre_root": out["root"].round(6)})
    res.to_csv(f"{P}/features/deepcre_soy_scores.tsv", sep="\t", index=False)
    cov = pd.read_csv(f"{P}/features/gene_covariates.tsv", sep="\t").set_index("gene").loc[res.gene]
    e = cov.expr_mean_log2tpm.values
    lo_q, hi_q = np.quantile(e, [0.25, 0.75])
    sel = (e <= lo_q) | (e >= hi_q)
    for t in ("leaf", "root"):
        print(f"check {t}: AUROC top vs bottom expression quartile {roc_auc_score(e[sel] >= hi_q, res['deepcre_' + t].values[sel]):.3f}; "
              f"Spearman with atlas mean {pd.Series(res['deepcre_' + t].values).corr(pd.Series(e), method='spearman'):.3f}")
    print(f"leaf-root Pearson {np.corrcoef(out['leaf'], out['root'])[0, 1]:.3f}")


def fit(split, labf, outp):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    sc = pd.read_csv(f"{P}/features/deepcre_soy_scores.tsv", sep="\t").set_index("gene")
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    Xall = np.column_stack([lg(sc.deepcre_leaf.values), lg(sc.deepcre_root.values)])
    gi = {g: i for i, g in enumerate(sc.index)}
    lab = pd.read_csv(labf, sep="\t")
    lab = lab[lab.stress.isin(STRESSES)].copy()
    lab["call"] = "unclear"
    lab.loc[(lab.vote == "up") & (lab.meta == "up"), "call"] = "up"
    lab.loc[(lab.vote == "down") & (lab.meta == "down"), "call"] = "down"
    lab.loc[(lab.meta == "nonresp") & ~lab.vote.isin(["up", "down"]), "call"] = "nonresp"
    sp = pd.read_csv(f"{P}/{split}", sep="\t")
    chrom = "role" in sp.columns
    preds, rows = [], []
    for s in STRESSES:
        ls = lab[lab.stress == s].set_index("gene").call
        for d in ("up", "down"):
            task = f"{s}-{d}"
            g = [x for x in ls.index if ls[x] in (d, "nonresp") and x in gi]
            y = np.array([1 if ls[x] == d else 0 for x in g])
            X = Xall[[gi[x] for x in g]]
            if chrom:
                units = []
                for r in sorted(sp["round"].unique()):
                    role = dict(zip(sp.gene[sp["round"] == r], sp.role[sp["round"] == r]))
                    units.append((r, np.array([role[x] == "train" for x in g]), np.array([role[x] == "test" for x in g])))
            else:
                fold = dict(zip(sp.gene, sp.fold))
                f = np.array([fold[x] for x in g])
                units = [(k, f != k, f == k) for k in range(5)]
            for k, tr, te in units:
                if y[te].sum() == 0:
                    continue
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                m = LogisticRegression(max_iter=1000).fit((X[tr] - mu) / sd, y[tr])
                p = m.predict_proba((X[te] - mu) / sd)[:, 1]
                yt = y[te]
                rows.append((task, k, int(te.sum()), int(yt.sum()), round(yt.mean(), 5),
                             round(average_precision_score(yt, p), 5), round(roc_auc_score(yt, p), 5)))
                preds.append(pd.DataFrame({"gene": np.array(g)[te], "task": task, "fold": k, "y": yt, "pred": p}))
    pr = pd.concat(preds)
    pr.to_csv(outp + "_predictions.tsv.gz", sep="\t", index=False)
    met = pd.DataFrame(rows, columns=["task", "fold", "n_test", "n_pos", "prevalence", "auprc", "auroc"])
    pooled = [(t, "pooled", len(q), int(q.y.sum()), round(q.y.mean(), 5), round(average_precision_score(q.y, q.pred), 5),
               round(roc_auc_score(q.y, q.pred), 5)) for t, q in pr.groupby("task")]
    met = pd.concat([met, pd.DataFrame(pooled, columns=met.columns)])
    met.to_csv(outp + "_metrics.tsv", sep="\t", index=False)
    s = met[met.fold == "pooled"].copy()
    s["lift"] = (s.auprc / s.prevalence).round(2)
    print(s.to_string(index=False))


if MODE == "score":
    score(*sys.argv[3:5])
else:
    fit(*sys.argv[3:6])
