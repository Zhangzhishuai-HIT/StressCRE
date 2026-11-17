"""Contribution of exact sites of known elements to StressCRE's convolutional network.

From the attribution files of attribution.py (work/attr_v1/<task>.npz: codes, hypothetical contributions, labels),
for the responsive genes of each task: actual contribution per base = hypothetical contribution of the observed base;
net contribution of a site = sum over its bases. Elements (either strand; the same patterns as element_enrichment.py,
ABRE = ACGTGG or ACGTGT): ABRE, G-box, HSE (three units), DRE/CRT, W-box, GCC-box, evening element.
Outputs
  results/element_contributions.tsv : per task and element: genes, sites, share of unmasked bases in sites, share of
                                      the positive contribution carried by site bases, ratio of the two, mean net
                                      contribution per site, share of sites with a positive net contribution, share of
                                      sites that share at least one base with an ABRE site (not for the ABRE itself).
  results/abre_contributions.tsv    : the ABRE and G-box rows of the same table (kept for the figure scripts).
  results/abre_contrib_position.tsv : per task, window half and 100-bp bin: ABRE sites and mean net contribution.
Window layout: index 0-1499 = TSS -1000..+499 (TSS at 1000); 1500-1519 = N; 1520-3019 = TTS -499..+1000 (TTS at 2019).
Usage: python -I abre_contrib.py <project>
"""
import re, sys
import numpy as np
import pandas as pd

P = sys.argv[1]
TASKS = [f"{s}-{d}" for s in ["drought", "salt", "heat", "cold", "alkaline"] for d in ("up", "down")]
ELEMENTS = {   # name: (pattern on either strand, site length)
    "ABRE": ("ACGTG[GT]|[AC]CACGT", 6),
    "G-box": ("CACGTG", 6),
    "HSE": ("GAA..TTC..GAA|TTC..GAA..TTC", 13),
    "DRE/CRT": ("[AG]CCGAC|GTCGG[CT]", 6),
    "W-box": ("TTGAC[CT]|[AG]GTCAA", 6),
    "GCC-box": ("AGCCGCC|GGCGGCT", 7),
    "EE": ("AAAATATCT|AGATATTTT", 9),
}
rows, prow = [], []
for t in TASKS:
    d = np.load(f"{P}/work/attr_v1/{t}.npz")
    pos = d["y"] == 1
    codes = d["codes"][pos]
    hyp = d["hyp"][pos].astype(np.float32)
    act = np.take_along_axis(hyp, np.minimum(codes, 3)[:, None, :].astype(np.int64), 1)[:, 0, :]
    act[codes > 3] = 0
    seqs = ["".join("ACGTN"[c] for c in row) for row in codes]
    tot_pos = float(np.clip(act, 0, None).sum())
    n_valid = int((codes < 4).sum())
    abre_rx, abre_mask = re.compile(f"(?=({ELEMENTS['ABRE'][0]}))"), []
    for s in seqs:
        mk = np.zeros(len(s), bool)
        for m in abre_rx.finditer(s):
            mk[m.start():m.start() + 6] = True
        abre_mask.append(mk)
    for el, (pat, L) in ELEMENTS.items():
        rx = re.compile(f"(?=({pat}))")
        sites, mask_pos, n_bases, n_ov = [], 0.0, 0, 0
        for i, s in enumerate(seqs):
            starts = sorted({m.start() for m in rx.finditer(s)})
            mask = np.zeros(len(s), bool)
            for p in starts:
                mask[p:p + L] = True
                sites.append((p, float(act[i, p:p + L].sum())))
                n_ov += bool(abre_mask[i][p:p + L].any())
            mask_pos += float(np.clip(act[i], 0, None)[mask].sum())
            n_bases += int(mask.sum())
        sp = np.array(sites) if sites else np.zeros((0, 2))
        rows.append(dict(task=t, element=el, genes=int(pos.sum()), sites=len(sp), base_share=n_bases / n_valid,
                         positive_contribution_share=mask_pos / tot_pos,
                         enrichment_of_contribution=(mask_pos / tot_pos) / (n_bases / n_valid) if n_bases else np.nan,
                         mean_net_contribution=sp[:, 1].mean() if len(sp) else np.nan,
                         share_sites_positive=(sp[:, 1] > 0).mean() if len(sp) else np.nan,
                         share_sites_overlapping_abre=n_ov / len(sp) if len(sp) and el != "ABRE" else np.nan))
        if el != "ABRE":
            continue
        c = sp[:, 0] + 3
        for half, sel, rel in (("TSS", c < 1500, c - 1000), ("TTS", c >= 1520, c - 2019)):
            b = (np.floor(rel[sel] / 100) * 100).astype(int)
            g = pd.DataFrame({"bin": b, "v": sp[sel, 1]}).groupby("bin").v.agg(["size", "mean"]).reset_index()
            for r in g.itertuples():
                prow.append(dict(task=t, half=half, bin_start=r.bin, sites=r.size, mean_net_contribution=r.mean))
    print(t, "done", flush=True)
out = pd.DataFrame(rows)
out.to_csv(f"{P}/results/element_contributions.tsv", sep="\t", index=False, float_format="%.5g")
out[out.element.isin(["ABRE", "G-box"])].drop(columns=["enrichment_of_contribution", "share_sites_overlapping_abre"]).to_csv(
    f"{P}/results/abre_contributions.tsv", sep="\t", index=False, float_format="%.5g")
pd.DataFrame(prow).to_csv(f"{P}/results/abre_contrib_position.tsv", sep="\t", index=False, float_format="%.5g")
pd.set_option("display.width", 200)
print(out[out.task.str.endswith("-up")].pivot(index="element", columns="task", values="positive_contribution_share").round(4).to_string())
