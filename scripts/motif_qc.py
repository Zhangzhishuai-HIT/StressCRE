"""QC of a STREME motif library from DAP-seq peaks (dapseq_motifs.py): Tomtom against JASPAR 2024 CORE plants.

Reports how many significant motifs (STREME E < 0.05) match a JASPAR plant motif (Tomtom q < 0.05; Pearson
distance, min overlap 5, no small-sample correction) and, per TF, the best JASPAR match of its top significant
motif, so that libraries built with different peak sets can be compared.

Usage: python -I motif_qc.py <project> <library_name> [<library_name> ...]
  library_name = features/<name>.meme / .tsv;  writes work/motif_qc/tomtom_<name>.tsv and results/motif_qc_<name>.tsv
"""
import collections, csv, os, subprocess, sys

P = sys.argv[1]
BIN = os.path.dirname(sys.executable)
J = "/home/user/zhangzhishuai/myhermes/down_data/JASPAR2024/JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt"
jname = {l.split()[1]: l.split()[2] for l in open(J) if l.startswith("MOTIF")}
os.makedirs(f"{P}/work/motif_qc", exist_ok=True)
for lib_name in sys.argv[2:]:
    out = f"{P}/work/motif_qc/tomtom_{lib_name}.tsv"
    with open(out, "w") as o:
        subprocess.run([os.path.join(BIN, "tomtom"), "-no-ssc", "-text", "-min-overlap", "5", "-dist", "pearson",
                        "-evalue", "-thresh", "10", f"{P}/features/{lib_name}.meme", J], stdout=o,
                       stderr=subprocess.DEVNULL, check=True)
    lib = {r["motif"]: r for r in csv.DictReader(open(f"{P}/features/{lib_name}.tsv"), delimiter="\t")}
    best = {}
    for r in csv.DictReader((l for l in open(out) if l.strip() and not l.startswith("#")), delimiter="\t"):
        q = float(r["q-value"])
        if r["Query_ID"] not in best or q < best[r["Query_ID"]][0]:
            best[r["Query_ID"]] = (q, jname.get(r["Target_ID"], r["Target_ID"]), r["Target_ID"])
    sig = [m for m, r in lib.items() if float(r["E"]) < 0.05]
    hit = [m for m in sig if m in best and best[m][0] < 0.05]
    top = collections.OrderedDict()
    for m in sig:
        top.setdefault(lib[m]["tf"], m)
    tfs = sorted({r["tf"] for r in lib.values()})
    rows = []
    for tf in tfs:
        m = top.get(tf)
        b = best.get(m, (1.0, "", "")) if m else (1.0, "", "")
        rows.append((tf, tf.split("_")[0], m or "", lib[m]["consensus"] if m else "", lib[m]["E"] if m else "",
                     b[1] if b[0] < 0.05 else "", b[2] if b[0] < 0.05 else "", f"{b[0]:.3g}" if m else ""))
    with open(f"{P}/results/motif_qc_{lib_name}.tsv", "w") as o:
        o.write("tf\tfamily\ttop_sig_motif\tconsensus\tE\tbest_jaspar_name\tbest_jaspar_id\tq\n")
        o.write("".join("\t".join(r) + "\n" for r in rows))
    n_top = sum(1 for r in rows if r[2])
    n_top_hit = sum(1 for r in rows if r[5])
    print(f"{lib_name}: {len(lib)} motifs; {len(sig)} with E<0.05, {len(hit)} of them match JASPAR plants (q<0.05); "
          f"TFs with a significant motif {n_top}/{len(tfs)}, whose top one matches JASPAR {n_top_hit}")
