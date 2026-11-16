#!/bin/bash
#SBATCH --job-name=scre_motif
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/baseline_motif.%j.log
# Merge the FIMO counts and run the random-forest motif baseline on the five splits (labels v1).
#   sbatch --dependency=afterok:<motif_scan array id> jobs/baseline_motif.sbatch
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin/python
LAB=${LAB:-$P/labels/labels_v1.tsv.gz}
echo "start $(date) $(hostname)"
$PY -I $P/scripts/motif_scan.py merge $P $P/work/motif 23
for x in "fold_seed0 main_seed0" "fold_seed1 main_seed1" "fold_seed2 main_seed2" "chrom_holdout chrom" "random_seed0 random"; do
  set -- $x
  echo "== $1"
  nice -n 10 $PY -I $P/scripts/baseline_motif.py $P splits/$1.tsv $LAB $P/results/v1_motif_$2 8
done
echo "end $(date)"
