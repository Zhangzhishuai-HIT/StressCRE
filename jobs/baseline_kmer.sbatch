#!/bin/bash
#SBATCH --job-name=scre_kmer
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/baseline_kmer.%x.%j.log
# k-mer LightGBM baseline on one split. Submit with:
#   sbatch -J scre_km0 --export=ALL,SPLIT=splits/fold_seed0.tsv,NAME=v1_kmer_main_seed0 jobs/baseline_kmer.sbatch
# (labels default to the frozen labels/labels_v1.tsv.gz; week-1 runs used LAB=labels/pilot_labels.tsv.gz; MaxRSS 2.4-2.8 GB)
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin/python
LAB=${LAB:-$P/labels/labels_v1.tsv.gz}
echo "start $(date) $(hostname) split=$SPLIT labels=$LAB"
nice -n 10 $PY -I $P/scripts/baseline_kmer.py $P $SPLIT $LAB $P/results/$NAME 8
echo "end $(date)"
