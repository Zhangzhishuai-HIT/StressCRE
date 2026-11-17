"""Heat-shock proteins and heat-shock transcription factors among heat-responsive genes.

Gene descriptions come from the Note field of the Wm82.a4 gene models (gff3). The protein description is the text
before the first semicolon; the rest lists InterPro domains and GO terms.
  heat-shock protein (HSP): protein description contains "heat shock" or "HSP" followed by a number (any case; so not
                            HSPRO2) and does not describe a binding, interacting or activating protein or a
                            transcription factor;
  heat-shock transcription factor (HSF): protein description contains "heat stress transcription factor",
                            "heat shock factor" or "HSF" (but not "binding"), or the domains include IPR027725 (heat
                            shock transcription factor family).
Heat-responsive genes are the strict calls used everywhere else (cre_data.strict_calls).
Usage: python -I heat_shock_genes.py <project>   -> results/heat_shock_genes.tsv (one row per HSP or HSF gene with a
       heat call) and a summary on stdout.
"""
import os, re, sys, urllib.parse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cre_data import strict_calls   # noqa: E402

P = sys.argv[1]
GFF = "/home/user/zhangzhishuai/myhermes/2080/SoyExpFM/ref/a4/genes.gff3"
HSP = re.compile(r"heat shock|\bhsp\d", re.I)
NOT_HSP = re.compile(r"binding|interact|activator|transcription factor|factor", re.I)
HSF_DESC = re.compile(r"heat stress transcription factor|heat shock factor|\bhsf", re.I)

note = {}
for line in open(GFF):
    f = line.rstrip("\n").split("\t")
    if len(f) < 9 or f[2] != "gene":
        continue
    a = dict(x.split("=", 1) for x in f[8].split(";") if "=" in x)
    note[a["Name"]] = urllib.parse.unquote(a.get("Note", ""))


def kind(g):
    n = note.get(g, "")
    d = n.split(";")[0]
    if (HSF_DESC.search(d) and not re.search("binding", d, re.I)) or "IPR027725" in n:
        return "HSF"
    if HSP.search(d) and not NOT_HSP.search(d):
        return "HSP"
    return ""


lab = strict_calls(pd.read_csv(f"{P}/labels/labels_v1.tsv.gz", sep="\t"))
q = lab[(lab.stress == "heat") & lab.call.isin(["up", "down", "nonresp"])][["gene", "call"]].drop_duplicates()
q["class"] = q.gene.map(kind)
q["description"] = q.gene.map(lambda g: note.get(g, "").split(";")[0])
out = q[q["class"] != ""].sort_values(["class", "call", "gene"])
out.to_csv(f"{P}/results/heat_shock_genes.tsv", sep="\t", index=False)
for c in ("HSP", "HSF"):
    for call in ("up", "down", "nonresp"):
        n_all = int((q.call == call).sum())
        n = int(((q.call == call) & (q["class"] == c)).sum())
        print(f"{c}\theat {call}\t{n} of {n_all}\t{100 * n / n_all:.2f}%")
print("HSF genes up-regulated by heat:", " ".join(out[(out["class"] == "HSF") & (out.call == "up")].gene))
