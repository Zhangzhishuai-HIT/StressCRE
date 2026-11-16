#!/bin/bash
#SBATCH --job-name=scre_lgbm
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/baseline_lgbm.%x.%j.log
# LightGBM baseline on one feature set and one split:
#   sbatch -J scre_cv0 --export=ALL,FEATURES=covariates,SPLIT=splits/fold_seed0.tsv,NAME=v1_cov_main_seed0 jobs/baseline_lgbm.sbatch
# (labels default to the frozen labels/labels_v1.tsv.gz; week-1 runs used LAB=labels/pilot_labels.tsv.gz; MaxRSS <=4 GB)
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin/python
LAB=${LAB:-$P/labels/labels_v1.tsv.gz}
echo "start $(date) $(hostname) features=$FEATURES split=$SPLIT"
nice -n 10 $PY -I $P/scripts/baseline_lgbm.py $P $SPLIT $LAB $P/results/$NAME 8
echo "end $(date)"
