"""Frozen splits for StressCRE (plan step 3). All methods use these files.

Connected groups (union-find over the 52,872 Wm82.a4 genes), edges from:
  (1) MMseqs2 30% protein families      work/fam/a4_c30_cluster.tsv
  (2) WGD paralog pairs                  results/wgd_pairs.tsv
  (3) window similarity: megablast hits between windows that do NOT overlap in the genome, >= 95% identity
      over >= 300 bp, both windows with <= 2 such alignments (rows of the hit table).
      Why so strict (measured 2026-11-10): at >= 80% / 200 bp, 120,636 window pairs at different loci
      are similar and chain 84% of all genes into one group; overlapping
      neighbour windows (identical sequence by position, 17,500 pairs) chain genes along chromosomes.
      Shared repeated sequence can be a regulatory signal the model may legitimately learn; homology leakage is
      handled by (1), (2) and CDS masking. Residual effect is reported as a diagnostic (test genes with
      a >= 90% / 300 bp promoter-similar training gene vs without).
Main split  : groups -> 5 folds, greedy balanced on genes and on pilot positives, seeds 0, 1, 2.
Chromosome  : 5 rounds, each holds out 4 of the 20 chromosomes (round k: Gm{k+1}, Gm{k+6}, Gm{k+11},
              Gm{k+16}); test genes with a homolog on a training chromosome are removed
              (homolog = WGD partner, promoter-similar window, or protein hit >= 50% identity).
              Scaffold genes are always training.
Random      : genes -> 5 folds at random (reference only).
If the largest group exceeds 5% of genes, (3) falls back to >= 98% / 300 bp / <= 3 partners, then to none.

Usage: python3 -I make_splits.py <project_dir>
"""
import collections, hashlib, os, sys
import numpy as np
import pandas as pd

P = sys.argv[1]
os.makedirs(f"{P}/splits", exist_ok=True)
win = pd.read_csv(f"{P}/seqs/wm82a4_windows.tsv", sep="\t")
genes = list(win.gene)
gi = {g: i for i, g in enumerate(genes)}
chrom = dict(zip(win.gene, win.chrom))
G = len(genes)
strip = lambda x: x.split("ann1.")[-1].rsplit(".", 1)[0]


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def f(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        a, b = self.f(a), self.f(b)
        if a != b:
            self.p[b] = a


fam = pd.read_csv(f"{P}/work/fam/a4_c30_cluster.tsv", sep="\t", header=None, names=["rep", "mem"])
fam_edges = [(gi[strip(r)], gi[strip(m)]) for r, m in zip(fam.rep, fam.mem) if strip(r) in gi and strip(m) in gi]
wgd = pd.read_csv(f"{P}/results/wgd_pairs.tsv", sep="\t")
wgd_edges = [(gi[a], gi[b]) for a, b in zip(wgd.gene_a, wgd.gene_b) if a in gi and b in gi]
ps = pd.read_csv(f"{P}/work/psim/a4_win_edges.m8", sep="\t", header=None,
                 names=["q", "t", "pid", "alnlen", "qs", "qe", "ts", "te", "ev", "bits"])
ps = ps[(ps.q != ps.t) & ps.q.isin(gi) & ps.t.isin(gi)]


def spans(r):
    if r.strand == "+":
        return [(r.tss - 1000, r.tss + 499), (r.tts - 499, r.tts + 1000)]
    return [(r.tss - 499, r.tss + 1000), (r.tts - 1000, r.tts + 499)]


SP = {r.gene: (r.chrom, spans(r)) for r in win.itertuples()}


def overlap(a, b):
    ca, sa = SP[a]; cb, sb = SP[b]
    return ca == cb and any(x1 <= y2 and y1 <= x2 for x1, x2 in sa for y1, y2 in sb)


ps = ps[[not overlap(a, b) for a, b in zip(ps.q, ps.t)]]


def promoter_edges(min_pid, min_len, cap):
    sel = ps[(ps.pid >= min_pid) & (ps.alnlen >= min_len)]
    if cap is not None:
        dg = collections.Counter(sel.q)
        sel = sel[sel.q.map(lambda x: dg[x] <= cap) & sel.t.map(lambda x: dg[x] <= cap)]
    return [(gi[q], gi[t]) for q, t in zip(sel.q, sel.t)]


def groups_with(pe):
    uf = UF(G)
    for a, b in fam_edges + wgd_edges + pe:
        uf.u(a, b)
    return np.array([uf.f(i) for i in range(G)])


for rule in [(95, 300, 2), (98, 300, 3), None]:
    pe = promoter_edges(*rule) if rule else []
    root = groups_with(pe)
    sizes = collections.Counter(root)
    if max(sizes.values()) <= 0.05 * G:
        break
    print(f"largest group {max(sizes.values())} > 5% with promoter rule {rule}; falling back")
n_ps = len(pe) // 2
promoter_rule = "none" if rule is None else f">= {rule[0]}% over >= {rule[1]} bp, <= {rule[2]} partners, non-overlapping"
# diagnostic set: any non-overlapping >= 90% / 300 bp window similarity (see similarity_diagnostic.py)
ps90 = ps[(ps.pid >= 90) & (ps.alnlen >= 300)]
te_like = collections.defaultdict(set)
for q, t in zip(ps90.q, ps90.t):
    te_like[gi[q]].add(gi[t])
gid_of = {r: k for k, (r, _) in enumerate(sizes.most_common())}
grp = np.array([gid_of[r] for r in root])
gsize = np.array([sizes[r] for r in root])
pd.DataFrame({"gene": genes, "group": grp, "group_size": gsize, "chrom": [chrom[g] for g in genes]}).to_csv(
    f"{P}/splits/groups.tsv", sep="\t", index=False)

# pilot positives per gene (for balancing only)
lab = pd.read_csv(f"{P}/labels/pilot_labels.tsv.gz", sep="\t")
lab = lab[lab.stress.isin(["drought", "salt", "heat", "cold", "alkaline"])]
lab["call"] = "unclear"
lab.loc[(lab.vote == "up") & (lab.meta == "up"), "call"] = "up"
lab.loc[(lab.vote == "down") & (lab.meta == "down"), "call"] = "down"
lab.loc[(lab.meta == "nonresp") & ~lab.vote.isin(["up", "down"]), "call"] = "nonresp"
pos = lab[lab.call.isin(["up", "down"])].groupby("gene").size()
gpos = np.array([pos.get(g, 0) for g in genes])
tasks = [(s, d) for s in ["drought", "salt", "heat", "cold", "alkaline"] for d in ("up", "down")]
task_pos = {t: set(lab.gene[(lab.stress == t[0]) & (lab.call == t[1])]) for t in tasks}

summary = [f"genes {G}; groups {len(sizes)}; largest {max(sizes.values())} "
           f"({100 * max(sizes.values()) / G:.2f}%); singletons {sum(1 for v in sizes.values() if v == 1)}; "
           f"promoter pairs used {n_ps} ({promoter_rule}); "
           f"family edges {len(fam_edges)}; WGD pairs {len(wgd_edges)}"]

# ---- main split: greedy balanced assignment of groups to 5 folds
gtab = pd.DataFrame({"group": grp, "pos": gpos}).groupby("group").agg(n=("pos", "size"), pos=("pos", "sum"))
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    order = gtab.sample(frac=1.0, random_state=seed).sort_values("n", ascending=False, kind="stable")
    load_n = np.zeros(5); load_p = np.zeros(5)
    fold_of = {}
    for g, r in order.iterrows():
        score = load_n / max(load_n.sum(), 1) + load_p / max(load_p.sum(), 1) + rng.uniform(0, 1e-6, 5)
        k = int(np.argmin(score))
        fold_of[g] = k; load_n[k] += r.n; load_p[k] += r.pos
    fold = np.array([fold_of[g] for g in grp])
    shared = [int(any(fold[j] != fold[i] for j in te_like.get(i, ()))) for i in range(G)]
    pd.DataFrame({"gene": genes, "fold": fold, "promoter_similar_in_other_fold": shared}).to_csv(
        f"{P}/splits/fold_seed{seed}.tsv", sep="\t", index=False)
    # self-checks
    cross = pd.DataFrame({"group": grp, "fold": fold}).groupby("group").fold.nunique().max()
    assert cross == 1, "a group spans two folds"
    per = {t: [len(task_pos[t] & set(np.array(genes)[fold == k])) for k in range(5)] for t in tasks}
    mn = min(min(v) for v in per.values())
    summary.append(f"seed {seed}: genes per fold {np.bincount(fold).tolist()}; min positives in any fold x task {mn}; "
                   + "; ".join(f"{t[0]}-{t[1]} {min(v)}–{max(v)}" for t, v in per.items()))

# ---- chromosome hold-out with homolog removal
hom = collections.defaultdict(set)
for a, b in wgd_edges:
    hom[a].add(b); hom[b].add(a)
for a, b in pe:
    hom[a].add(b); hom[b].add(a)
for line in open(f"{P}/work/fam/a4_self.m8"):
    f = line.split("\t")
    a, b = strip(f[0]), strip(f[1])
    if a != b and float(f[2]) >= (50 if float(f[2]) > 1 else 0.5) and a in gi and b in gi:
        hom[gi[a]].add(gi[b]); hom[gi[b]].add(gi[a])
chrs = [f"glyma.Wm82.gnm4.Gm{k:02d}" for k in range(1, 21)]
rows = []
for k in range(5):
    held = {chrs[k], chrs[k + 5], chrs[k + 10], chrs[k + 15]}
    test = [i for i, g in enumerate(genes) if chrom[g] in held]
    train = set(i for i, g in enumerate(genes) if chrom[g] not in held)
    kept = [i for i in test if not (hom[i] & train)]
    keep = set(kept)
    for i, g in enumerate(genes):
        role = "train" if i in train else ("test" if i in keep else "removed")
        rows.append((g, k, role))
    npos = {t: len(task_pos[t] & set(genes[i] for i in kept)) for t in tasks}
    summary.append(f"chrom round {k} ({', '.join(sorted(h.split('.')[-1] for h in held))}): test {len(test)}, "
                   f"kept after homolog removal {len(kept)}; positives kept " +
                   ", ".join(f"{t[0]}-{t[1]} {v}" for t, v in npos.items()))
pd.DataFrame(rows, columns=["gene", "round", "role"]).to_csv(f"{P}/splits/chrom_holdout.tsv", sep="\t", index=False)

# ---- random reference split
rf = np.random.default_rng(0).integers(0, 5, G)
pd.DataFrame({"gene": genes, "fold": rf}).to_csv(f"{P}/splits/random_seed0.tsv", sep="\t", index=False)

open(f"{P}/splits/SPLITS_SUMMARY.txt", "w").write("\n".join(summary) + "\n")
with open(f"{P}/splits/SHA256SUMS", "w") as o:
    for fn in sorted(os.listdir(f"{P}/splits")):
        if fn.endswith(".tsv"):
            o.write(hashlib.sha256(open(f"{P}/splits/{fn}", "rb").read()).hexdigest() + "  " + fn + "\n")
print("\n".join(summary))
