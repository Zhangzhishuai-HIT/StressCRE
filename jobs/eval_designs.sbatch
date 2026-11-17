#!/bin/bash
#SBATCH --job-name=scre_evd
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/eval_designs.%j.log
# Paired bootstrap (1,000 resamples of gene groups, expression-matched AUPRC) of StressCRE against gene features + k-mers
# in the chromosome hold-out and random designs.   sbatch jobs/eval_designs.sbatch   (OUT=<dir> to write elsewhere)
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin/python
OUT=${OUT:-$P/results}
mkdir -p $OUT
echo "start $(date) $(hostname)"
for d in chrom random; do
  nice -n 10 $PY -I $P/scripts/evaluate.py $P $P/results/v1_cand3_${d}_predictions.tsv.gz $P/results/v1_stack_${d}_predictions.tsv.gz \
    --matched --n 1000 --out $OUT/eval_v1_cand3_vs_stack_${d}.tsv > /dev/null &
done
wait
echo "end $(date)"
