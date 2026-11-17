"""Robustness of the accuracy comparison to window similarity that remains between training and test genes.

Window similarity: megablast hits between windows of different genes that do not overlap in the genome
(work/psim/a4_win_edges.m8, the hits used by make_splits.py) with >= 90% identity over >= 300 bp. Most such windows
share transposon fragments and were deliberately not grouped (make_splits.py groups only windows with at most two
alignments at >= 95%).
  Main design: test genes flagged in splits/fold_seed<s>.tsv (promoter_similar_in_other_fold = such a window in a gene
               of another fold) are removed from the test predictions of StressCRE and of gene features + k-mers.
  Chromosome hold-out: kept test genes with such a window in a gene on a training chromosome of their round are
               removed.
Writes the filtered prediction files to work/simdiag/ (same columns as results/*_predictions.tsv.gz) and the numbers
of removed genes to results/similarity_diagnostic_counts.tsv; jobs/similarity_diagnostic.sbatch then runs evaluate.py
on them (results/eval_v1_cand3_vs_stack_nosim_<design>.tsv) and summarises the paired comparisons in
results/similarity_diagnostic.tsv (--summary).

Usage: python -I similarity_diagnostic.py <project>              (filter)
       python -I similarity_diagnostic.py <project> --summary    (after the evaluate.py runs)
"""
import os, sys
import pandas as pd

P = sys.argv[1]
W = f"{P}/work/simdiag"
os.makedirs(W, exist_ok=True)
MAIN9 = [f"{s}-{d}" for s in ["drought", "salt", "heat", "cold", "alkaline"] for d in ("up", "down")][:-1]

if "--summary" in sys.argv:
    rows = []
    for design in ["main_seed0", "main_seed1", "main_seed2", "chrom"]:
        e = pd.read_csv(f"{P}/results/eval_v1_cand3_vs_stack_nosim_{design}.tsv", sep="\t")
        e9 = e[e.task.isin(MAIN9)]
        rows.append(dict(design=design, mean_matched_auprc_stresscre=round(e9.auprc_a.mean(), 4),
                         mean_matched_auprc_stack=round(e9.auprc_b.mean(), 4), tasks_higher=int((e9["diff"] > 0).sum()),
                         tasks_significant=",".join(e9[e9.verdict == "A better"].task),
                         tasks_significantly_lower=",".join(e9[e9.verdict == "B better"].task)))
    out = pd.DataFrame(rows)
    out.to_csv(f"{P}/results/similarity_diagnostic.tsv", sep="\t", index=False)
    print(out.to_string(index=False))
    sys.exit(0)

win = pd.read_csv(f"{P}/seqs/wm82a4_windows.tsv", sep="\t")
chrom = dict(zip(win.gene, win.chrom))


def spans(r):
    if r.strand == "+":
        return [(r.tss - 1000, r.tss + 499), (r.tts - 499, r.tts + 1000)]
    return [(r.tss - 499, r.tss + 1000), (r.tts - 1000, r.tts + 499)]


SP = {r.gene: (r.chrom, spans(r)) for r in win.itertuples()}


def overlap(a, b):
    ca, sa = SP[a]; cb, sb = SP[b]
    return ca == cb and any(x1 <= y2 and y1 <= x2 for x1, x2 in sa for y1, y2 in sb)


ps = pd.read_csv(f"{P}/work/psim/a4_win_edges.m8", sep="\t", header=None,
                 names=["q", "t", "pid", "alnlen", "qs", "qe", "ts", "te", "ev", "bits"])
ps = ps[(ps.q != ps.t) & ps.q.isin(SP) & ps.t.isin(SP) & (ps.pid >= 90) & (ps.alnlen >= 300)]
ps = ps[[not overlap(a, b) for a, b in zip(ps.q, ps.t)]]
partners = {}
for a, b in zip(ps.q, ps.t):
    partners.setdefault(a, set()).add(b)
    partners.setdefault(b, set()).add(a)

counts = []
for s in range(3):
    f = pd.read_csv(f"{P}/splits/fold_seed{s}.tsv", sep="\t")
    flagged = set(f.gene[f.promoter_similar_in_other_fold == 1])
    for m in ("cand3", "stack"):
        p = pd.read_csv(f"{P}/results/v1_{m}_main_seed{s}_predictions.tsv.gz", sep="\t")
        p[~p.gene.isin(flagged)].to_csv(f"{W}/{m}_main_seed{s}_predictions.tsv.gz", sep="\t", index=False)
    counts.append(dict(design=f"main_seed{s}", genes_removed=len(flagged), genes=len(f)))

ch = pd.read_csv(f"{P}/splits/chrom_holdout.tsv", sep="\t")
removed = set()
for k, g in ch.groupby("round"):
    train = set(g.gene[g.role == "train"])
    for gene in g.gene[g.role == "test"]:
        if partners.get(gene, set()) & train:
            removed.add((k, gene))
for m in ("cand3", "stack"):
    p = pd.read_csv(f"{P}/results/v1_{m}_chrom_predictions.tsv.gz", sep="\t")
    keep = [(fo, g) not in removed for fo, g in zip(p.fold, p.gene)]
    p[keep].to_csv(f"{W}/{m}_chrom_predictions.tsv.gz", sep="\t", index=False)
counts.append(dict(design="chrom", genes_removed=len(removed), genes=int((ch.role == "test").sum())))
pd.DataFrame(counts).to_csv(f"{P}/results/similarity_diagnostic_counts.tsv", sep="\t", index=False)
print(pd.DataFrame(counts).to_string(index=False))
