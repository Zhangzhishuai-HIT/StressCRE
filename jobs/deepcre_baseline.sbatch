#!/bin/bash
#SBATCH --job-name=scre_dcre
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/deepcre_baseline.%j.log
# DeepCRE soybean leaf/root models (rebuilt in PyTorch from the Keras h5) as a baseline: score every gene,
# then a logistic regression on the two scores per task for the five splits (labels v1).
#   sbatch jobs/deepcre_baseline.sbatch
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin/python
G=/home/user/zhangzhishuai/myhermes/down_data/LIS_genomes_annotations/Glycine_max/Wm82.gnm4.4PTR/glyma.Wm82.gnm4.4PTR.genome_main.fna.gz
H=/home/user/zhangzhishuai/myhermes/down_data/GitHub_NAMlab_deepCRE_soybean
LAB=${LAB:-$P/labels/labels_v1.tsv.gz}
echo "start $(date) $(hostname)"
[ -s $P/features/deepcre_soy_scores.tsv ] || nice -n 10 $PY -I $P/scripts/deepcre_baseline.py score $P $G $H
for x in "fold_seed0 main_seed0" "fold_seed1 main_seed1" "fold_seed2 main_seed2" "chrom_holdout chrom" "random_seed0 random"; do
  set -- $x
  echo "== $1"
  nice -n 10 $PY -I $P/scripts/deepcre_baseline.py fit $P splits/$1.tsv $LAB $P/results/v1_deepcre_$2
done
echo "end $(date)"
