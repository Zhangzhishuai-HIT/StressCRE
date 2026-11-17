"""Stress-vs-control contrasts in the SoyExpFM sample table (shared by the StressCRE scripts).

A contrast is one (study, cultivar, organ group, genetic background) cell with at least
`min_per_arm` treated runs and `min_per_arm` control runs:
  treated = role 'treated', the stress is among its treatments, and no other abiotic stress is;
  control = role 'control' with the same cultivar, organ group and genetic background.
Runs whose role is 'unclear' are never used, nor transgenic, edited, mutant or VIGS lines: only
backgrounds 'none' and 'near-isogenic or introgression line' (ALLOWED_GM). By default only
Glycine max studies are used (species=None for all): wild-soybean studies are kept for the
cross-genome test.
Iron deficiency has no treatment term of its own (SoyExpFM files it under 'nutrient'), so it is
'nutrient' within the studies whose curated design mentions iron (iron_studies()).
"""
import json, os
import pandas as pd

SX = "/home/user/zhangzhishuai/myhermes/2080/SoyExpFM"
ABIO = ["drought", "salt", "heat", "cold", "alkaline stress", "flooding or submergence",
        "osmotic stress", "heavy metal"]
SHORT = {"drought": "drought", "salt": "salt", "heat": "heat", "cold": "cold",
         "alkaline stress": "alkaline", "flooding or submergence": "flooding",
         "osmotic stress": "osmotic", "heavy metal": "metal", "iron deficiency": "iron"}
NUT = "nutrient (N, P, K, Fe, S or other)"
ALLOWED_GM = {"none", "near-isogenic or introgression line"}


def iron_studies():
    out = set()
    for line in open(f"{SX}/metadata/curation/designs.jsonl"):
        d = json.loads(line)
        t = " ".join([d.get("design", ""), d.get("notes", ""), d.get("sample_rule", "")]).lower()
        if NUT in d.get("treatment_types", []) and ("iron" in t or " fe " in t or "fe-" in t or "fe deficien" in t):
            out.add(d["study"])
    return out


# role corrections: labels/role_overrides.tsv in the project, config/role_overrides.tsv in the code repository
OVERRIDES = [f for f in ("/home/user/zhangzhishuai/myhermes/2080/StressCRE/labels/role_overrides.tsv",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "role_overrides.tsv"))
             if os.path.exists(f)]


def load_annotation(overrides=True):
    """SoyExpFM sample table, with StressCRE's own role corrections (labels/role_overrides.tsv: runs whose
    SoyExpFM role was 'unclear' but whose ENA sample alias / submitted file name gives the design)."""
    a = pd.read_csv(f"{SX}/metadata/sample_annotation.tsv", sep="\t", dtype=str).fillna("")
    if overrides:
        if not OVERRIDES:
            raise FileNotFoundError("role_overrides.tsv not found (labels/ or config/); the labels need these corrections")
        ov = pd.read_csv(OVERRIDES[0], sep="\t", dtype=str).set_index("run")
        hit = a.run.isin(ov.index)
        for col in ("role", "cultivar", "organ_group", "genetic_modification", "treatments"):
            a.loc[hit, col] = a.loc[hit, "run"].map(ov[col])
    q = pd.read_csv(f"{SX}/data/expression/runs.tsv", sep="\t", dtype=str)
    a["quantified"] = a.run.isin(set(q.run))
    a["tl"] = a.treatments.map(lambda t: [x.strip() for x in t.split(";") if x.strip()])
    return a


def find_contrasts(a, stress, min_per_arm=2, quantified_only=True, species="Glycine max"):
    if stress == "iron deficiency":
        fe = iron_studies()
        is_tr = (a.role == "treated") & a.study.isin(fe) & a.tl.map(lambda l: NUT in l and not (set(l) & set(ABIO)))
    else:
        others = set(ABIO) - {stress}
        is_tr = (a.role == "treated") & a.tl.map(lambda l: stress in l and not (set(l) & others))
    studies = set(a[is_tr].study)
    d = a[a.study.isin(studies) & a.genetic_modification.isin(ALLOWED_GM)]
    if species:
        d = d[d.species == species]
    if quantified_only:
        d = d[d.quantified]
    out = []
    for study, g in d.groupby("study"):
        tr = g[is_tr.loc[g.index]]
        co = g[g.role == "control"]
        for (cv, org, gm), t2 in tr.groupby(["cultivar", "organ_group", "genetic_modification"]):
            c2 = co[(co.cultivar == cv) & (co.organ_group == org) & (co.genetic_modification == gm)]
            if len(t2) >= min_per_arm and len(c2) >= min_per_arm:
                cid = f"{SHORT[stress]}__{study}__{len(out) + 1:03d}"
                out.append({
                    "contrast": cid, "stress": SHORT[stress], "study": study,
                    "cultivar": cv, "organ_group": org, "genetic_modification": gm,
                    "species": g.species.iloc[0],
                    "treated": list(t2.run), "control": list(c2.run),
                    "time_treated": list(t2.hours_after_treatment),
                    "time_control": list(c2.hours_after_treatment),
                    "tissue": ",".join(sorted(set(t2.tissue) | set(c2.tissue))),
                })
    return out
