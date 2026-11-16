#!/bin/bash
#SBATCH --job-name=scre_eval
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/eval_candidates.%j.log
# Paired bootstrap (1,000 resamples of connected groups, expression-matched AUPRC) of model A against the
# references, per seed of the main split.   sbatch --export=ALL,A=cand3,REFS="stack cov kmer" jobs/eval_candidates.sbatch
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin/python
A=${A:-cand3}; REFS=${REFS:-stack cov}
echo "start $(date) $(hostname) A=$A refs=$REFS"
for s in 0 1 2; do
  for r in $REFS; do
    nice -n 10 $PY -I $P/scripts/evaluate.py $P $P/results/v1_${A}_main_seed${s}_predictions.tsv.gz \
      $P/results/v1_${r}_main_seed${s}_predictions.tsv.gz --matched --n 1000 --out $P/results/eval_v1_${A}_vs_${r}_seed${s}.tsv > /dev/null &
  done
  wait
done
echo "end $(date)"
