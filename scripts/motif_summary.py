"""Summary of the learned motifs of the up-regulation tasks (the numbers of the Results on motif discovery).

From results/modisco_patterns.tsv (modisco_motifs.py). Per motif: summed contribution = sites x mean contribution per site
over the trimmed positions (cwm_sum). A motif is ABRE-like when its consensus contains ACGTG on either strand (ACGTG or
CACGT) and heat-shock-element-like (HSE-like) when it contains GAAnnTTC or TTCnnGAA (the two orientations of a two-unit
HSE); case is ignored. Per up-regulation task:
  first motif by summed contribution (consensus, sites, summed contribution, share of its sites with an ACGTG core),
  the first motif whose consensus lacks the ACGTG core and the ratio of the two summed contributions,
  the rank by sites of the first motif among motifs with a positive mean contribution, and the motif with most sites,
  the HSE-like motifs with a positive mean contribution (number, sites, summed contribution, ratio to the first motif).
Usage: python -I motif_summary.py <project>    -> results/motif_summary.tsv
"""
import sys
import pandas as pd

P = sys.argv[1]
mp = pd.read_csv(f"{P}/results/modisco_patterns.tsv", sep="\t")
mp["total"] = mp.n_seqlets * mp.cwm_sum
cons = mp.consensus.str.upper()
mp["abre_like"] = cons.str.contains("ACGTG|CACGT")
mp["hse_like"] = cons.str.contains("GAA..TTC|TTC..GAA")
rows = []
for t in ["drought-up", "salt-up", "heat-up", "cold-up", "alkaline-up"]:
    q = mp[mp.task == t].sort_values("total", ascending=False)
    first = q.iloc[0]
    other = q[~q.abre_like].iloc[0]
    pos = q[q.cwm_sum > 0].sort_values("n_seqlets", ascending=False)
    hse = q[q.hse_like & (q.cwm_sum > 0)]
    rows.append(dict(task=t, first_consensus=first.consensus, first_abre_like=bool(first.abre_like), first_sites=int(first.n_seqlets),
                     first_total=round(first.total, 2), first_share_acgtg_core=first.share_acgtg_core,
                     next_without_core=other.consensus, next_total=round(other.total, 2), ratio=round(first.total / other.total, 2),
                     first_rank_by_sites=int(list(pos.index).index(first.name) + 1), most_sites_consensus=pos.iloc[0].consensus,
                     most_sites=int(pos.iloc[0].n_seqlets), hse_like_motifs=len(hse), hse_like_sites=int(hse.n_seqlets.sum()),
                     hse_like_total=round(hse.total.sum(), 2), hse_like_ratio_to_first=round(hse.total.sum() / first.total, 3)))
out = pd.DataFrame(rows)
out.to_csv(f"{P}/results/motif_summary.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False))
