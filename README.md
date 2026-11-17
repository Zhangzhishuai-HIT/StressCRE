# StressCRE

Code for the manuscript *StressCRE: a sequence model of soybean responses to five abiotic stresses identifies the ABA-responsive element as a shared activating cis-element*.

StressCRE predicts whether a soybean (*Glycine max*, Wm82.a4) gene is up-regulated, down-regulated or non-responsive under drought, salt, heat, cold and alkaline stress (ten tasks). It reads a 3,020-bp window around each gene (1,000 bp upstream to 500 bp downstream of the transcription start site, 20 N, 500 bp upstream to 1,000 bp downstream of the transcript end; coding sequence masked) with two networks:

- a convolutional network trained from scratch on the one-hot window;
- a network on frozen representations of the DNA language model PlantCAD2-Small (50-bp bins, attention pooling per task).

Both networks also receive 15 gene features (atlas expression level, breadth and tissue specificity, gene structure, window GC content, family size and duplication). StressCRE predicts the mean of the two networks' probabilities. Models are compared with an expression-matched AUPRC, in which non-responsive test genes are reweighted to the expression level and tissue specificity of the responsive genes.

The repository contains code only. Input data are public (see below); labels, predictions, motifs and SNP effects are in the Supporting Information of the article.

## Layout

| Folder | Content |
|---|---|
| `scripts/` | Python and R scripts of every step (listed below) |
| `scripts/figures/` | figures of the article (`figstyle.py` holds the shared style) |
| `jobs/` | SLURM job scripts and the environment set-up (`setup_env.sh`, `setup_env_gpu.sh`) |
| `config/` | consensus sequences of the elements used in the insertion experiments (`insertion_elements.tsv`) and the corrections of run roles (`role_overrides.tsv`) |

Paths at the top of the scripts point to the authors' cluster; set them to your own copies of the input data.

## Steps

| Step | Scripts |
|---|---|
| Stress-versus-control comparisons, DESeq2 per comparison | `contrasts.py`, `inventory_contrasts.py`, `de_prepare.py`, `de_run.R`, `jobs/pilot_de.sbatch` |
| Labels (vote over studies and random-effects meta-analysis) | `pilot_labels.py` (`pilot_labels.py work/de_v1 <project> v1` writes `labels/labels_v1.tsv.gz`) |
| Sequence windows and gene features | `make_windows.py`, `check_windows.py`, `gene_features.py`, `kmer_features.py`, `embed_plantcad2.py`, `fix_embed_chunk_rows.py` |
| Gene groups and folds (protein families, WGD pairs, similar windows) | `jobs/protein_families.sbatch`, `paralog_pairs.py`, `jobs/promoter_sim.sbatch`, `make_splits.py` |
| StressCRE | `cre_models.py`, `train_cre.py`, `cre_data.py` |
| Comparison models | `baseline_kmer.py`, `baseline_lgbm.py`, `stack_oof.py`, `motif_scan.py`, `baseline_motif.py`, `deepcre_baseline.py`, `koh_baseline.py` |
| Accuracy (expression-matched AUPRC, paired bootstrap over gene groups) | `cre_eval.py`, `evaluate.py`, `summarize_models.py`, `week2_summary.py` (means over tasks and test designs), `jobs/eval_candidates.sbatch`, `jobs/eval_designs.sbatch` (chromosome and random designs), `similarity_diagnostic.py` and `jobs/similarity_diagnostic.sbatch` (without test genes sharing a similar window) |
| Contribution scores, motifs and element insertion | `attribution.py`, `modisco_prep.py`, `jobs/modisco.sbatch`, `modisco_motifs.py`, `dapseq_motifs.py`, `motif_qc.py`, `element_enrichment.py`, `abre_contrib.py`, `heat_shock_genes.py`, `insertion.py`, `insertion_logodds.py`, `jobs/insertion_runs.sh` |
| WGD paralog pairs | `predict_all.py`, `paralog_analysis.py`, `paralog_motifs.py` |
| Natural variants | `variant_sites.py`, `jobs/variant_af.sbatch`, `score_variants.py`, `variant_analysis.py` |
| Wild soybean W05 | `w05_orthologs.py`, `w05_homwindows.py`, `predict_w05.py`, `w05_analysis.py` |
| External files (direct downloads with checksums) | `fetch_external.py` |

## Requirements

Python 3.10 with PyTorch 2.5.1, transformers 4.46.3, mamba_ssm and causal_conv1d (for PlantCAD2), PEFT 0.18.1 (for the adapters of Koh et al. 2026), LightGBM 4.7.0, scikit-learn, statsmodels, NumPy, pandas, SciPy, h5py, edlib, numba and tfmodisco-lite 3.0; R 4.3.3 with DESeq2 1.42.0; MEME suite 5.5.9 (FIMO, Tomtom, STREME), MMseqs2 18, BLAST+, DIAMOND and bcftools. `jobs/setup_env.sh` creates the Python environment (with BLAST+, the MEME suite, TF-MoDISco-lite, statsmodels, edlib and PEFT); `jobs/setup_env_gpu.sh` adds PyTorch and transformers and copies prebuilt mamba_ssm 2.2.4 and causal_conv1d 1.5.0.post8 from another local environment (installing them with pip works as well). MMseqs2, DIAMOND, bcftools and R with DESeq2 came from separate installations. `train_cre.py` gained the `--features` option after the main models were trained (their `run.json` records the earlier file); its default (`gene`) is the code path used for them. GPU steps (PlantCAD2 representations, training, attribution, insertion, SNP scoring) were run on NVIDIA RTX 4090 GPUs.

## Input data (public)

- RNA-seq runs of the stress studies (NCBI SRA, DDBJ and ENA; BioProject accessions in Table S1 of the article) and of the Plant Gene Atlas of Williams 82 (Sreedasyam et al. 2023), processed with fastp and salmon on the Wm82.a4 transcripts.
- Run annotation (role, cultivar, organ, treatment and hours after treatment) from a curation of the public run metadata; `contrasts.py` reads it as a table, with the corrections in `config/role_overrides.tsv`. Table S1 of the article lists every comparison with its treated and control runs.
- Wm82.a4 genome and gene models (Valliyodan et al. 2019), and the Wm82.a2 genome and gene models (Song et al. 2016) for the SNP windows.
- PlantCAD2-Small weights (Hugging Face, kuleshov-group).
- Soybean DAP-seq peaks of 148 transcription factors (Jiao et al. 2024, SoyTFBase) and JASPAR 2024 CORE plants motifs.
- DeepCRE soybean models (Peleke et al. 2024) and the released adapters of Koh et al. (2026).
- SNPs of 1,511 resequenced accessions (Zhang et al. 2022; Legume Information System, Wm82.gnm2.div.Zhang_Jiang_2020). The 300,000 SNP–gene combinations scored with StressCRE (150,000 at random and 150,000 with a minor allele frequency of at least 5%) are listed in Data S2 of the article.
- *G. soja* W05 genome (Xie et al. 2019) and RNA-seq (BioProject PRJNA521577).

## License

MIT (see `LICENSE`).
