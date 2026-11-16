"""Cross-species reference baseline (plan step 5, table 4): Koh et al. 2026 (Mol Plant; Kingdom Stress Atlas) PlantCAD2
LoRA adapters, applied as released to soybean genes.

Weights: figshare 10.6084/m9.figshare.31995939, Supplementary Dataset 5 (down_data/Koh2026_KingdomStress_weights/
dataset5/figshare_archive/plantcad2_adapters/<Stress>_<UP|DOWN>/), one binary classifier per stress x direction:
PEFT LoRA r=8 (alpha 32) on the Mamba in/x/out projections of kuleshov-group/PlantCAD2-Small-l24-d0768 plus the
`score` head of the model's own CaduceusForSequenceClassification (mean pooling over both strands, 2 logits).
Input as in their code (github mutwil/KingdomStress Figure_6/04_plantcad2_finetune.py, extract_dna_strings):
600 bp centred on the TSS (300 bp upstream + 300 bp from the TSS), transcript orientation, coding sequence NOT
masked, default tokenizer call. TSS = primary-transcript start (seqs/wm82a4_windows.tsv).
Tasks covered: drought, salt, heat, cold x up/down (no alkaline model). Their training data included soybean
(140 comparisons, ~70% of orthogroups in training), so scores on our genes are optimistic: reference only.

Usage: python -I koh_baseline.py windows <project> <genome.fa.gz>        -> seqs/wm82a4_tss600_koh.fa.gz (cluster)
       CUDA_VISIBLE_DEVICES=g python -I koh_baseline.py score <project> <model_dir> <adapter_root> [batch=64] [Drought_UP,...]
         -> features/koh2026/<Stress>_<UP|DOWN>.tsv (GPU; only the genes labelled for that task; existing files skipped)
       python -I koh_baseline.py predictions <project>           -> results/v1_koh_<split>_predictions.tsv.gz
         (same genes, tasks, folds and labels as results/v1_cov_<split>_predictions.tsv.gz; pred = Koh score)
"""
import gzip, os, sys, time
import numpy as np
import pandas as pd

MODE, P = sys.argv[1:3]
TASKS = {"drought": "Drought", "salt": "Salt", "heat": "Heat", "cold": "Cold"}

if MODE == "windows":
    fa = sys.argv[3]
    genome, name, buf = {}, None, []
    for line in gzip.open(fa, "rt"):
        if line.startswith(">"):
            if name:
                genome[name] = "".join(buf).upper()
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    genome[name] = "".join(buf).upper()
    comp = str.maketrans("ACGTN", "TGCAN")
    win = pd.read_csv(f"{P}/seqs/wm82a4_windows.tsv", sep="\t")
    n_skip = 0
    with gzip.open(f"{P}/seqs/wm82a4_tss600_koh.fa.gz", "wt") as o:
        for r in win.itertuples():
            L = len(genome[r.chrom])
            if r.strand == "+":
                s, e = r.tss - 1 - 300, r.tss - 1 + 300          # 0-based half-open: 300 up + 300 from the TSS
            else:
                s, e = r.tss - 300, r.tss + 300                  # minus strand: TSS is the 1-based end coordinate
            if s < 0 or e > L:
                n_skip += 1
                continue
            seq = genome[r.chrom][s:e]
            if r.strand == "-":
                seq = seq.translate(comp)[::-1]
            o.write(f">{r.gene}\n{seq}\n")
    print(f"{len(win) - n_skip} windows of 600 bp; {n_skip} skipped (too close to a chromosome end)")

elif MODE == "score":
    import torch
    try:
        import torch.distributed.tensor  # noqa: F401  (PEFT 0.18 looks up torch.distributed.tensor.DTensor; torch 2.5 needs the import)
    except ImportError:
        pass
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel
    from safetensors.torch import load_file
    md, ad = sys.argv[3], sys.argv[4]
    B = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    subset = sys.argv[6].split(",") if len(sys.argv) > 6 else [f"{v}_{d}" for v in TASKS.values() for d in ("UP", "DOWN")]
    names, seqs = [], []
    for line in gzip.open(f"{P}/seqs/wm82a4_tss600_koh.fa.gz", "rt"):
        if line.startswith(">"):
            names.append(line[1:].strip())
        else:
            seqs.append(line.strip())
    gi = {g: i for i, g in enumerate(names)}
    # only genes labelled for the task are scored (strict up/down vs non-responsive, labels v1 rule)
    lab = pd.read_csv(f"{P}/labels/labels_v1.tsv.gz", sep="\t")
    lab["call"] = "unclear"
    lab.loc[(lab.vote == "up") & (lab.meta == "up"), "call"] = "up"
    lab.loc[(lab.vote == "down") & (lab.meta == "down"), "call"] = "down"
    lab.loc[(lab.meta == "nonresp") & ~lab.vote.isin(["up", "down"]), "call"] = "nonresp"
    inv = {v: k for k, v in TASKS.items()}
    tok = AutoTokenizer.from_pretrained(md, trust_remote_code=True)
    print(f"{len(names)} windows; tokens per window {len(tok(seqs[0])['input_ids'])} (default tokenizer call)", flush=True)
    os.makedirs(f"{P}/features/koh2026", exist_ok=True)
    t0 = time.time()
    for ad_name in subset:
        fo = f"{P}/features/koh2026/{ad_name}.tsv"
        if os.path.exists(fo):
            continue
        st, d = ad_name.split("_")
        q = lab[(lab.stress == inv[st]) & lab.call.isin([d.lower(), "nonresp"]) & lab.gene.isin(gi)]
        idx = sorted(gi[g] for g in q.gene)
        a = os.path.join(ad, ad_name)
        base = AutoModelForSequenceClassification.from_pretrained(md, trust_remote_code=True, num_labels=2)
        model = PeftModel.from_pretrained(base, a).cuda().eval()
        w = load_file(os.path.join(a, "adapter_model.safetensors"))["base_model.model.score.weight"]
        got = model.base_model.model.score.modules_to_save["default"].weight.detach().cpu()
        assert torch.allclose(got, w), "score head not loaded"
        inner = model.get_base_model()
        assert sum(1 for n_, _ in inner.named_modules() if n_.endswith(".lora_A")) > 0, "no LoRA layers injected"
        p = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(idx), B):
                ids = tok([seqs[i] for i in idx[s:s + B]], return_tensors="pt")["input_ids"].cuda()
                # the LoRA-injected inner model: PeftModelForSequenceClassification.forward passes attention_mask,
                # which CaduceusForSequenceClassification.forward does not accept
                p.append(torch.softmax(inner(input_ids=ids).logits.float(), -1)[:, 1].cpu().numpy())
        pd.DataFrame({"gene": [names[i] for i in idx], "score": np.concatenate(p).round(6)}).to_csv(fo + ".tmp", sep="\t", index=False)
        os.replace(fo + ".tmp", fo)
        print(f"{ad_name}: {len(idx)} labelled genes scored  [{time.time() - t0:.0f}s]", flush=True)
        del model, base, inner
        torch.cuda.empty_cache()

else:
    sc = {f.split("/")[-1][:-4]: pd.read_csv(f, sep="\t").set_index("gene").score
          for f in sorted(__import__("glob").glob(f"{P}/features/koh2026/*_*.tsv"))}
    for split in ["main_seed0", "main_seed1", "main_seed2", "chrom", "random"]:
        ref = pd.read_csv(f"{P}/results/v1_cov_{split}_predictions.tsv.gz", sep="\t")
        parts = []
        for task, q in ref.groupby("task"):
            s, d = task.split("-")
            if s not in TASKS:
                continue
            col = f"{TASKS[s]}_{d.upper()}"
            if col not in sc:
                continue
            q = q[q.gene.isin(sc[col].index)].copy()
            q["pred"] = sc[col].loc[q.gene].values
            parts.append(q)
        pr = pd.concat(parts)
        pr.to_csv(f"{P}/results/v1_koh_{split}_predictions.tsv.gz", sep="\t", index=False)
        print(split, len(pr), "rows,", pr.task.nunique(), "tasks")
