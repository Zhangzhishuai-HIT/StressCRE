#!/bin/bash
#SBATCH --job-name=scre_fam
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/protein_families.%j.log
# Protein families and the all-against-all protein search (MMseqs2 18.8cc5c) used by gene_features.py, make_splits.py
# and paralog_pairs.py:
#   work/fam/a4_c30_cluster.tsv  families: clusters at >= 30% identity covering >= 50% of each member protein
#   work/fam/a4_self.m8          hits at >= 50% identity covering >= 50% of both proteins (at most 40 per query)
#   sbatch jobs/protein_families.sbatch              (OUT=<dir> sbatch --export=ALL,OUT=<dir> ... to write elsewhere)
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
MM=${MMSEQS:-/home/user/zhangzhishuai/.local/share/mamba/envs/thermoplm/bin/mmseqs}
FAA=/home/user/zhangzhishuai/myhermes/down_data/LIS_genomes_annotations/Glycine_max/Wm82.gnm4.ann1.T8TQ/glyma.Wm82.gnm4.ann1.T8TQ.protein_primary.faa.gz
OUT=${OUT:-$P/work/fam}
mkdir -p $OUT && cd $OUT
echo "start $(date) $(hostname) -> $OUT"
zcat $FAA | sed 's/\*$//' > a4.faa
$MM easy-cluster a4.faa a4_c30 tmp --min-seq-id 0.30 -c 0.5 --cov-mode 1 --threads 6 -v 1
$MM easy-search a4.faa a4.faa a4_self.m8 tmp2 --min-seq-id 0.5 -c 0.5 --max-seqs 40 --threads 6 -v 1 \
    --format-output query,target,pident,alnlen,evalue,bits,qcov,tcov
rm -rf tmp tmp2
echo "end $(date)"
