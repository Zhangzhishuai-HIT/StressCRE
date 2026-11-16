"""Inputs for tfmodisco-lite from attribution.py output: one-hot and hypothetical contribution arrays of a task's
positive (responsive) genes, shape [n, 4, L] float32 (modisco-lite CLI format).

Usage: python -I modisco_prep.py <attr_dir> <task> <out_dir>
"""
import os, sys
import numpy as np

A, T, O = sys.argv[1:4]
d = np.load(os.path.join(A, f"{T}.npz"))
pos = d["y"] == 1
codes = d["codes"][pos]
oh = np.zeros((len(codes), 4, codes.shape[1]), np.float32)
for b in range(4):
    oh[:, b] = codes == b
os.makedirs(O, exist_ok=True)
np.save(os.path.join(O, "ohe.npy"), oh)
np.save(os.path.join(O, "hyp.npy"), d["hyp"][pos].astype(np.float32))
with open(os.path.join(O, "genes.txt"), "w") as f:
    f.write("\n".join(d["genes"][pos]) + "\n")
print(T, oh.shape)
