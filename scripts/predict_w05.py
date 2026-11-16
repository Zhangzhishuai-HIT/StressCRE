"""StressCRE predictions from wild-soybean W05 windows (cross-genome test, plan step 8).

For every one-to-one W05 - Wm82.a4 ortholog pair, the W05 window anchored at the homologous TSS and TTS
(seqs/w05_homwindows.fa.gz from w05_homwindows.py; headers = Wm82.a4 gene ids; same layout and coding-sequence
masking as the a4 windows) is scored with the Wm82 gene's own gene features and the
models of the three split seeds in which the Wm82 gene was in the test fold; the Wm82-window prediction of the same
gene with the same features is in results/v1_stresscre_allgenes.tsv.gz, so the two predictions differ only by
sequence. PlantCAD2 representations of the W05 windows: embed_plantcad2.py on the same FASTA.

Usage: CUDA_VISIBLE_DEVICES=<g> python -I predict_w05.py <project> <w05_homwindows.fa.gz> <w05_homwindows.tsv> <embedding_dir>
Writes results/w05_predictions.tsv.gz (gene_a4, gene_w05, task, p_w05, p_w05_cnn, p_w05_pc2).
"""
import glob, os, sys, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_models import TASKS, encode, onehot, revcomp_onehot   # noqa: E402
from cre_data import read_windows, covariates, zscore, load_fold_model   # noqa: E402

P, WFA, ORTH, EMB = sys.argv[1:5]
dev = torch.device("cuda")
t0 = time.time()
names4, _ = read_windows(f"{P}/seqs/wm82a4_windows.fa.gz")
gi4 = {g: i for i, g in enumerate(names4)}
C = covariates(f"{P}/features/gene_covariates.tsv", names4)
folds = {s: pd.read_csv(f"{P}/splits/fold_seed{s}.tsv", sep="\t").set_index("gene").fold.reindex(names4).values for s in range(3)}
namesw, seqsw = read_windows(WFA)
giw = {g: i for i, g in enumerate(namesw)}
chunks = sorted(glob.glob(os.path.join(EMB, "chunk_????.npy")))
assert sum((open(c[:-4] + ".genes.txt").read().split() for c in chunks), []) == namesw, "W05 embedding order"
E = np.concatenate([np.load(c) for c in chunks])
orth = pd.read_csv(ORTH, sep="\t")
orth = orth[orth.gene_a4.isin(gi4) & orth.gene_a4.isin(giw)].reset_index(drop=True)
i4 = orth.gene_a4.map(gi4).values
iw = orth.gene_a4.map(giw).values                          # window headers are the Wm82.a4 gene ids
codes = torch.from_numpy(np.stack([encode(seqsw[i]) for i in iw])).to(dev)
print(f"{len(orth)} ortholog pairs; inputs loaded in {time.time() - t0:.0f}s", flush=True)
out = {k: np.zeros((len(orth), len(TASKS)), np.float32) for k in ("cnn", "pc2")}
for kind in ("cnn", "pc2"):
    for s in range(3):
        for k in range(5):
            model, mu, sd, _ = load_fold_model(f"{P}/models/v1_{kind}_main_seed{s}/fold{k}.pt", C.shape[1], dev)
            rows = np.where(folds[s][i4] == k)[0]
            for b0 in range(0, len(rows), 256):
                rb = rows[b0:b0 + 256]
                c = torch.from_numpy(zscore(C[i4[rb]], mu, sd)).to(dev)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if kind == "cnn":
                        x = onehot(codes[torch.from_numpy(rb).to(dev)])
                        z = 0.5 * (model(x, c).float() + model(revcomp_onehot(x), c).float())
                    else:
                        z = model(torch.from_numpy(E[iw[rb]]).to(dev).float(), c).float()
                out[kind][rb] += torch.sigmoid(z).cpu().numpy() / 3
    print(kind, "done", flush=True)
rows = []
for j, t in enumerate(TASKS):
    rows.append(pd.DataFrame({"gene_a4": orth.gene_a4, "gene_w05": orth.gene_w05, "task": t,
                              "p_w05": ((out["cnn"][:, j] + out["pc2"][:, j]) / 2).round(6),
                              "p_w05_cnn": out["cnn"][:, j].round(6), "p_w05_pc2": out["pc2"][:, j].round(6)}))
pd.concat(rows).to_csv(f"{P}/results/w05_predictions.tsv.gz", sep="\t", index=False)
print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
