"""Contribution of exact ABRE sites (ACGTGG or ACGTGT, either strand) to StressCRE's convolutional network.

From the attribution files of attribution.py (work/attr_v1/<task>.npz: codes, hypothetical contributions, labels),
for the responsive genes of each task: actual contribution per base = hypothetical contribution of the observed base;
net contribution of a site = sum over its 6 bases.
Outputs
  results/abre_contributions.tsv : per task: genes, sites, share of unmasked bases in sites, share of the positive
                                   contribution carried by site bases, mean net contribution per site, share of sites
                                   with a positive net contribution; the same for G-box (CACGTG) sites.
  results/abre_contrib_position.tsv : per task, window half and 100-bp bin: sites and mean net contribution per site.
Window layout: index 0-1499 = TSS -1000..+499 (TSS at 1000); 1500-1519 = N; 1520-3019 = TTS -499..+1000 (TTS at 2019).
Usage: python -I abre_contrib.py <project>
"""
import re, sys
import numpy as np
import pandas as pd

P = sys.argv[1]
TASKS = [f"{s}-{d}" for s in ["drought", "salt", "heat", "cold", "alkaline"] for d in ("up", "down")]
ELEMENTS = {"ABRE": ("ACGTG[GT]", "[AC]CACGT"), "G-box": ("CACGTG", None)}   # forward pattern, reverse-strand pattern
rows, prow = [], []
for t in TASKS:
    d = np.load(f"{P}/work/attr_v1/{t}.npz")
    pos = d["y"] == 1
    codes = d["codes"][pos]
    hyp = d["hyp"][pos].astype(np.float32)
    act = np.take_along_axis(hyp, np.minimum(codes, 3)[:, None, :].astype(np.int64), 1)[:, 0, :]
    act[codes > 3] = 0
    for el, (fw, rv) in ELEMENTS.items():
        sites, mask_pos, tot_pos, n_bases, n_valid = [], 0.0, 0.0, 0, 0
        for i in range(len(codes)):
            s = "".join("ACGTN"[c] for c in codes[i])
            starts = {m.start() for m in re.finditer(f"(?={fw})", s)}
            if rv:
                starts |= {m.start() for m in re.finditer(f"(?={rv})", s)}
            mask = np.zeros(len(s), bool)
            for p in starts:
                mask[p:p + 6] = True
                sites.append((p, float(act[i, p:p + 6].sum())))
            a = np.clip(act[i], 0, None)
            tot_pos += a.sum(); mask_pos += a[mask].sum(); n_bases += mask.sum(); n_valid += int((codes[i] < 4).sum())
        sp = np.array(sites)
        rows.append(dict(task=t, element=el, genes=int(pos.sum()), sites=len(sp), base_share=n_bases / n_valid,
                         positive_contribution_share=mask_pos / tot_pos, mean_net_contribution=sp[:, 1].mean(),
                         share_sites_positive=(sp[:, 1] > 0).mean()))
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
out.to_csv(f"{P}/results/abre_contributions.tsv", sep="\t", index=False, float_format="%.5g")
pd.DataFrame(prow).to_csv(f"{P}/results/abre_contrib_position.tsv", sep="\t", index=False, float_format="%.5g")
print(out.round(4).to_string(index=False))
