"""Soybean TF motifs from DAP-seq peaks (Jiao et al. 2024, SoyTFBase; 148 TFs, a4 coordinates) for plan step 6:
the peaks come without position weight matrices, so motifs are re-derived with STREME (MEME suite 5.5.9).

Per TF: up to 1,000 peaks drawn at random (seed 0) from its peaks on the 20 chromosomes (201-bp windows; peaks on
unplaced scaffolds are skipped because their names do not match the genome file), sequences from Wm82.a4;
STREME --dna --nmotifs 3 --minw 6 --maxw 15 against its default control (shuffled primary sequences, 2-mer
frequencies kept). Motifs are renamed <family>_<Glyma ID>_<k> and merged into one MEME file for Tomtom.

Usage: python -I dapseq_motifs.py extract <project> <genome.fa.gz> <peak_dir> <work_dir> [n_peaks=1000] [width=0]
         (width 0 = the whole 201-bp peak window; width w = the central w bp)
       python -I dapseq_motifs.py merge <project> <work_dir> [out_name=soy_dapseq_streme]
         -> features/<out_name>.meme, .tsv
(STREME runs between the two steps: jobs/dapseq_motifs.sbatch)
"""
import glob, gzip, os, re, sys
import numpy as np

MODE, P = sys.argv[1:3]
if MODE == "extract":
    fa, peak_dir, work = sys.argv[3:6]
    NP = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
    WD = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    genome, name, buf = {}, None, []
    for line in gzip.open(fa, "rt"):
        if line.startswith(">"):
            if name:
                genome[name] = "".join(buf).upper()
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    genome[name] = "".join(buf).upper()
    short = {k.split(".")[-1]: k for k in genome}          # Gm01 -> glyma.Wm82.gnm4.Gm01
    os.makedirs(f"{work}/fa", exist_ok=True)
    for f in sorted(glob.glob(f"{peak_dir}/*_peak.bed")):
        tf = os.path.basename(f)[:-len("_peak.bed")]
        rows = [l.split()[:3] for l in open(f) if l.strip() and not l.startswith(("#", "track"))]
        n_all = len(rows)
        rows = [r for r in rows if r[0] in short]           # peaks on unplaced scaffolds (e.g. Gm15_scaffold_113,
                                                            # 13,263 of 2,985,530) have no matching sequence name
        rng = np.random.default_rng(0)
        pick = sorted(rng.choice(len(rows), min(NP, len(rows)), replace=False))
        n = 0
        with open(f"{work}/fa/{tf}.fa", "w") as o:
            for i in pick:
                c, s, e = rows[i]
                s, e = int(s), int(e)
                if WD:
                    s = (s + e) // 2 - WD // 2
                    e = s + WD
                seq = genome[short[c]][s:e]
                if len(seq) >= min(50, WD or 50) and seq.count("N") < 0.1 * len(seq):
                    o.write(f">{c}:{s}-{e}\n{seq}\n"); n += 1
        print(f"{tf}\t{n_all} peaks\t{len(rows)} on chromosomes\t{n} used", flush=True)
else:
    work = sys.argv[3]
    ON = sys.argv[4] if len(sys.argv) > 4 else "soy_dapseq_streme"
    out, tab, head = [], [], None
    for d in sorted(glob.glob(f"{work}/streme/*/streme.txt")):
        tf = d.split("/")[-2]
        txt = open(d).read()
        if head is None:
            head = txt[:txt.index("MOTIF")]
        blocks = [b for b in re.split(r"(?m)^(?=MOTIF )", txt[txt.index("MOTIF"):]) if b.startswith("MOTIF ")]
        for k, b in enumerate(blocks, 1):
            first = b.split("\n", 1)
            m = re.match(r"MOTIF (\S+) (\S+)", first[0])
            ev = re.search(r"E= (\S+)", b)
            nsites = re.search(r"nsites= (\S+)", b)
            nm = f"{tf}_{k}"
            out.append(f"MOTIF {nm} {m.group(1)}\n" + first[1].rstrip() + "\n\n")
            tab.append((nm, tf, m.group(1), ev.group(1) if ev else "", nsites.group(1) if nsites else ""))
    with open(f"{P}/features/{ON}.meme", "w") as o:
        o.write(head + "".join(out))
    with open(f"{P}/features/{ON}.tsv", "w") as o:
        o.write("motif\ttf\tconsensus\tE\tnsites\n" + "".join("\t".join(r) + "\n" for r in tab))
    print(f"{len(out)} motifs from {len(set(r[1] for r in tab))} TFs")
