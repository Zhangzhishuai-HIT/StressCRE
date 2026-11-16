"""Train one StressCRE candidate on one split (plan step 4).

Candidates (networks in cre_models.py)
  cnn  candidate 1: DeepCRE-style CNN from scratch on the 3,020-bp one-hot window; random reverse complement in
       training, forward/reverse-complement average at prediction.
  pc2  candidate 2: head on frozen PlantCAD2-Small features (embeddings/plantcad2_small_wm82a4, 60 bins x 1,536).
Both take the 15 gene covariates (features/gene_covariates.tsv; lengths log1p, z-scored on the training genes of
each outer fold, missing -> 0, plus a tau-missing flag).

Tasks and labels as in baseline_kmer.py: task <stress>-<up|down>; positive = strict up (down) = vote and
meta-analysis agree; negative = non-responsive; every other gene is masked out of that task.
Loss = mean over tasks of the task's mean binary cross-entropy (masked entries dropped).

Protocol for every outer test fold k (as for the LightGBM baselines):
  1. train on the training folds minus the inner fold (k+1)%5 (chromosome hold-out: minus a random 20% of the
     training genes); score the inner fold after every epoch with the mean expression-matched AUPRC over the 9
     main tasks (alkaline-down is secondary; weights as in evaluate.py --matched). Constant learning rate for the
     first --min-epochs epochs (the inner-fold score sits on a plateau for ~10 epochs before the sequence part
     starts to add, so stopping earlier cuts runs short), then the learning rate halves after --plateau epochs
     without improvement (floor 1e-5) and training stops after --patience epochs without improvement;
  2. refit on all training folds for the best number of epochs with the same seed and the same per-epoch
     learning rates, and predict fold k.
  (A first launch with patience 6 and no minimum stopped some runs on that plateau; it was discarded before
  any test-fold result was read: logs/aborted_patience6/.)
Step-1 predictions on the inner folds (one per gene across the 5 outer folds) are saved: the choice between
candidates is made on them, inside the training folds; the test folds are used once (plan 2.4).

Usage: CUDA_VISIBLE_DEVICES=<g> python -I train_cre.py --project P --model cnn|pc2 --split splits/fold_seed0.tsv
         --name v1_cnn_main_seed0 [--labels labels/labels_v1.tsv.gz] [--seed 0] [--folds 0,1,2,3,4]
         [--epochs 80] [--min-epochs 20] [--patience 10] [--plateau 4] [--lr ...] [--wd ...] [--bs 128] [--drop ...] [--mem-frac 0.25]
Writes results/<name>_predictions.tsv.gz and _innerval.tsv.gz (gene, task, fold, y, pred), results/<name>_metrics.tsv,
models/<name>/fold<k>.pt (refit model) and models/<name>/run.json (versions, sha256 of inputs, hyper-parameters,
per-epoch history, best epochs, inner-fold and test scores).
"""
import argparse, datetime, glob, gzip, hashlib, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_models import CNN, PC2Head, TASKS, encode, onehot, revcomp_onehot   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True); ap.add_argument("--model", choices=["cnn", "pc2"], required=True)
ap.add_argument("--split", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--labels", default="labels/labels_v1.tsv.gz")
ap.add_argument("--windows", default="seqs/wm82a4_windows.fa.gz")
ap.add_argument("--emb", default="embeddings/plantcad2_small_wm82a4")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--folds", default="")
ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--min-epochs", type=int, default=20); ap.add_argument("--plateau", type=int, default=4)
ap.add_argument("--lr", type=float); ap.add_argument("--wd", type=float); ap.add_argument("--drop", type=float)
ap.add_argument("--bs", type=int, default=128); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--mem-frac", type=float, default=0.25)
ap.add_argument("--features", choices=["gene", "none"], default="gene",
                help="none: gene features replaced by zeros (sequence-only model, for comparison with sequence-only methods)")
a = ap.parse_args()
P = a.project
HP = dict(cnn=dict(lr=1e-3, wd=1e-4, drop=0.25), pc2=dict(lr=3e-4, wd=1e-2, drop=0.2))[a.model]
for k in ("lr", "wd", "drop"):
    if getattr(a, k) is not None:
        HP[k] = getattr(a, k)
MAIN = [j for j, t in enumerate(TASKS) if t != "alkaline-down"]
pj = lambda x: x if os.path.isabs(x) else os.path.join(P, x)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.set_num_threads(a.threads)
assert torch.cuda.is_available()
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(a.mem_frac)
t_start = time.time()

# ---- genes (window order) and inputs
names, seqs = [], []
for line in gzip.open(pj(a.windows), "rt"):
    if line.startswith(">"):
        names.append(line[1:].strip())
    else:
        seqs.append(line.strip())
N = len(names)
gi = {g: i for i, g in enumerate(names)}
if a.model == "cnn":
    X = torch.from_numpy(np.stack([encode(s) for s in seqs])).to(dev)          # uint8 [N, 3020], ~160 MB
    del seqs
else:
    del seqs
    chunks = sorted(glob.glob(os.path.join(pj(a.emb), "chunk_????.npy")))
    eg = sum((open(c[:-4] + ".genes.txt").read().split() for c in chunks), [])
    assert eg == names, "embedding gene order differs from the windows FASTA"
    X, r0 = None, 0
    for c in chunks:                                                           # float16 [N, 60, 1536], ~9 GB RAM
        z = np.load(c, mmap_mode="r")
        if X is None:
            X = np.empty((N,) + z.shape[1:], np.float16)
        X[r0:r0 + len(z)] = z
        r0 += len(z)
    assert r0 == N
print(f"{N} genes; model {a.model}; inputs {tuple(X.shape)} loaded in {time.time() - t_start:.0f}s", flush=True)

# ---- labels: Y[N, 10] in {1, 0, -1 = masked}
lab = pd.read_csv(pj(a.labels), sep="\t")
lab = lab[lab.stress.isin({t.split("-")[0] for t in TASKS})].copy()
lab["call"] = "unclear"
lab.loc[(lab.vote == "up") & (lab.meta == "up"), "call"] = "up"
lab.loc[(lab.vote == "down") & (lab.meta == "down"), "call"] = "down"
lab.loc[(lab.meta == "nonresp") & ~lab.vote.isin(["up", "down"]), "call"] = "nonresp"
Y = np.full((N, len(TASKS)), -1, np.int8)
for j, t in enumerate(TASKS):
    s, d = t.split("-")
    q = lab[(lab.stress == s) & lab.call.isin([d, "nonresp"]) & lab.gene.isin(set(names))]
    Y[q.gene.map(gi).values, j] = (q.call == d).astype(np.int8).values
print("labelled genes per task (pos/neg):",
      ", ".join(f"{t} {(Y[:, j] == 1).sum()}/{(Y[:, j] == 0).sum()}" for j, t in enumerate(TASKS)), flush=True)

# ---- covariates
cov = pd.read_csv(pj("features/gene_covariates.tsv"), sep="\t").set_index("gene").loc[names]
C = cov.copy()
for c in ["tx_len", "cds_len", "n_exons", "utr5_len", "utr3_len", "intron_len", "family_size"]:
    C[c] = np.log1p(C[c])
C["tau_missing"] = cov.expr_tau.isna().astype(float)
C = C.values.astype(np.float32)
if a.features == "none":
    C = np.zeros_like(C)
E_MEAN, E_TAU = cov.expr_mean_log2tpm.values, cov.expr_tau.values


def match_weights(idx, y):
    """evaluate.py --matched: negatives reweighted to the positives' (expression level x tau) bin mix."""
    e, t = E_MEAN[idx], E_TAU[idx]
    eb = pd.qcut(pd.Series(e).rank(method="first"), 5, labels=False).values
    tb = np.where(np.isnan(t), -1, pd.qcut(pd.Series(np.nan_to_num(t, nan=-1)).rank(method="first"), 5, labels=False).values)
    tb = np.where(np.isnan(t), 5, tb)
    b = eb * 10 + tb
    w = np.zeros(len(y))
    for k in np.unique(b):
        m = b == k
        npos, nneg = y[m].sum(), (1 - y[m]).sum()
        if nneg == 0:
            continue
        w[m & (y == 1)] = 1.0
        w[m & (y == 0)] = (npos / max(y.sum(), 1)) / (nneg / max((1 - y).sum(), 1)) if npos > 0 else 0.0
    return w


# ---- split units: (k, train_inner, val_inner, train_all, test) boolean masks over genes
sp = pd.read_csv(pj(a.split), sep="\t").set_index("gene")
units = []
if "role" in sp.columns:
    for r in sorted(sp["round"].unique()):
        role = sp[sp["round"] == r].role.reindex(names).fillna("none").values
        tr, te = role == "train", role == "test"
        va = tr & (np.random.default_rng(r).random(N) < 0.2)
        units.append((int(r), tr & ~va, va, tr, te))
else:
    f = sp.fold.reindex(names).values
    for k in range(5):
        te, va = f == k, f == (k + 1) % 5
        units.append((k, ~te & ~va & ~np.isnan(f), va, ~te & ~np.isnan(f), te))
if a.folds:
    keep = {int(x) for x in a.folds.split(",")}
    units = [u for u in units if u[0] in keep]

Yt = torch.from_numpy(Y).to(dev)


def fetch(idx):
    """inputs for gene indices idx (sorted numpy array) -> device tensors."""
    c = torch.from_numpy(Cz[idx]).to(dev)
    if a.model == "cnn":
        return onehot(X[torch.from_numpy(idx).to(dev)]), c
    return torch.from_numpy(X[idx]).to(dev, non_blocking=True).float(), c


def logits(model, x, c, train):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if a.model == "cnn":
            if train:
                flip = torch.rand(x.shape[0], device=dev) < 0.5
                x = torch.where(flip[:, None, None], revcomp_onehot(x), x)
                return model(x, c).float()
            return 0.5 * (model(x, c).float() + model(revcomp_onehot(x), c).float())
        return model(x, c).float()


def mt_loss(z, y):
    m = y >= 0
    l = F.binary_cross_entropy_with_logits(z, y.clamp(min=0).float(), reduction="none")
    per = (l * m).sum(0) / m.sum(0).clamp(min=1)
    return per[m.any(0)].mean()


def train_epoch(model, opt, idx, rng):
    model.train()
    perm = rng.permutation(idx)
    tot, nb = 0.0, 0
    for s in range(0, len(perm), a.bs):
        b = np.sort(perm[s:s + a.bs])
        if len(b) < 2:
            continue
        x, c = fetch(b)
        loss = mt_loss(logits(model, x, c, True), Yt[torch.from_numpy(b).to(dev)])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += loss.item(); nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def predict(model, idx):
    model.eval()
    out = []
    for s in range(0, len(idx), 512):
        x, c = fetch(idx[s:s + 512])
        out.append(torch.sigmoid(logits(model, x, c, False)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TASKS)), np.float32)


def new_model():
    torch.manual_seed(a.seed)
    m = (CNN(C.shape[1], len(TASKS), HP["drop"]) if a.model == "cnn" else PC2Head(C.shape[1], len(TASKS), drop=HP["drop"])).to(dev)
    return m, torch.optim.AdamW(m.parameters(), lr=HP["lr"], weight_decay=HP["wd"])


def rows_of(idx, p, k):
    out = []
    for j, t in enumerate(TASKS):
        m = Y[idx, j] >= 0
        out.append(pd.DataFrame({"gene": [names[i] for i in idx[m]], "task": t, "fold": k,
                                 "y": Y[idx[m], j].astype(int), "pred": p[m, j]}))
    return pd.concat(out)


os.makedirs(pj(f"models/{a.name}"), exist_ok=True)
os.makedirs(pj("results"), exist_ok=True)
run = dict(name=a.name, model=a.model, features=a.features, hyper=dict(HP, bs=a.bs, epochs_max=a.epochs, min_epochs=a.min_epochs, patience=a.patience,
                                                 plateau=a.plateau, lr_factor=0.5, lr_min=1e-5, seed=a.seed),
           split=a.split, split_sha256=sha256(pj(a.split)), labels=a.labels, labels_sha256=sha256(pj(a.labels)),
           covariates_sha256=sha256(pj("features/gene_covariates.tsv")),
           script_sha256=sha256(os.path.abspath(__file__)),
           models_sha256=sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cre_models.py")),
           torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
           started=datetime.datetime.now().isoformat(timespec="seconds"), folds={})
if a.model == "pc2":
    run["embeddings"] = a.emb
test_rows, val_rows = [], []
for k, tr_i, va_i, tr_all, te in units:
    t0 = time.time()
    lab_any = (Y >= 0).any(1)
    tr_idx, va_idx = np.where(tr_i & lab_any)[0], np.where(va_i & lab_any)[0]
    all_idx, te_idx = np.where(tr_all & lab_any)[0], np.where(te & lab_any)[0]
    # covariates z-scored on this fold's training genes
    mu, sd = np.nanmean(C[tr_all], 0), np.nanstd(C[tr_all], 0) + 1e-6
    Cz = np.nan_to_num((C - mu) / sd).astype(np.float32)
    vw = {j: match_weights(va_idx[Y[va_idx, j] >= 0], Y[va_idx[Y[va_idx, j] >= 0], j]) for j in MAIN}

    def val_score(p):
        sc = []
        for j in MAIN:
            m = Y[va_idx, j] >= 0
            sc.append(average_precision_score(Y[va_idx[m], j], p[m, j], sample_weight=vw[j]))
        return float(np.mean(sc)), sc

    model, opt = new_model()
    rng = np.random.default_rng(a.seed * 100 + k)
    hist, best, best_ep, best_p, wait, bad, lr, lrs = [], -1.0, 0, None, 0, 0, HP["lr"], []
    for ep in range(1, a.epochs + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr
        lrs.append(lr)
        tl = train_epoch(model, opt, tr_idx, rng)
        pv = predict(model, va_idx)
        sc, _ = val_score(pv)
        hist.append(dict(epoch=ep, lr=lr, train_loss=round(tl, 5), val_matched_auprc=round(sc, 5)))
        print(f"fold {k} ep {ep}: lr {lr:.1e} loss {tl:.4f} inner-val matched AUPRC (9 tasks) {sc:.4f}  [{time.time() - t0:.0f}s]", flush=True)
        if sc > best:
            best, best_ep, best_p, wait, bad = sc, ep, pv, 0, 0
        else:
            wait += 1; bad += 1
        if ep >= a.min_epochs:
            if bad >= a.plateau:
                lr, bad = max(lr * 0.5, 1e-5), 0
            if wait >= a.patience:
                break
    val_rows.append(rows_of(va_idx, best_p, k))
    _, per_task = val_score(best_p)
    # refit on all training folds for best_ep epochs
    model, opt = new_model()
    rng = np.random.default_rng(a.seed * 100 + k)
    for ep in range(best_ep):
        for gr in opt.param_groups:
            gr["lr"] = lrs[ep]
        train_epoch(model, opt, all_idx, rng)
    torch.save(dict(state_dict=model.state_dict(), cov_mu=mu, cov_sd=sd, hyper=run["hyper"], model=a.model),
               pj(f"models/{a.name}/fold{k}.pt"))
    pt = predict(model, te_idx)
    test_rows.append(rows_of(te_idx, pt, k))
    run["folds"][str(k)] = dict(best_epoch=best_ep, inner_val_matched_auprc_mean9=round(best, 5),
                                inner_val_matched_auprc={TASKS[j]: round(v, 5) for j, v in zip(MAIN, per_task)},
                                n_train_inner=len(tr_idx), n_val_inner=len(va_idx), n_train_all=len(all_idx),
                                n_test=len(te_idx), seconds=round(time.time() - t0), history=hist)
    print(f"fold {k}: best epoch {best_ep}, inner-val {best:.4f}; refit on {len(all_idx)} genes; "
          f"test predictions for {len(te_idx)} genes  [{time.time() - t0:.0f}s]", flush=True)

pr, pv = pd.concat(test_rows), pd.concat(val_rows)
pr.to_csv(pj(f"results/{a.name}_predictions.tsv.gz"), sep="\t", index=False)
pv.to_csv(pj(f"results/{a.name}_innerval.tsv.gz"), sep="\t", index=False)
met = []
for (task, k), q in pr.groupby(["task", "fold"]):
    if q.y.sum() > 0 and (q.y == 0).sum() > 0:
        met.append((task, k, len(q), int(q.y.sum()), round(q.y.mean(), 5),
                    round(average_precision_score(q.y, q.pred), 5), round(roc_auc_score(q.y, q.pred), 5)))
for task, q in pr.groupby("task"):
    met.append((task, "pooled", len(q), int(q.y.sum()), round(q.y.mean(), 5),
                round(average_precision_score(q.y, q.pred), 5), round(roc_auc_score(q.y, q.pred), 5)))
met = pd.DataFrame(met, columns=["task", "fold", "n_test", "n_pos", "prevalence", "auprc", "auroc"])
met.to_csv(pj(f"results/{a.name}_metrics.tsv"), sep="\t", index=False)
run["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
run["seconds"] = round(time.time() - t_start)
run["test_pooled"] = {r.task: dict(auprc=r.auprc, auroc=r.auroc, prevalence=r.prevalence)
                      for r in met[met.fold == "pooled"].itertuples()}
json.dump(run, open(pj(f"models/{a.name}/run.json"), "w"), indent=1)
s = met[met.fold == "pooled"].copy()
s["lift"] = (s.auprc / s.prevalence).round(2)
print(s.to_string(index=False))
print(f"done in {(time.time() - t_start) / 60:.1f} min", flush=True)
