#!/bin/bash
#SBATCH --job-name=scre_simd
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/similarity_diagnostic.%j.log
# StressCRE vs gene features + k-mers without the test genes that keep a similar window (>= 90% over >= 300 bp) in
# another fold or on a training chromosome (scripts/similarity_diagnostic.py): paired bootstrap, 1,000 resamples.
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin/python
W=$P/work/simdiag
echo "start $(date) $(hostname)"
$PY -I $P/scripts/similarity_diagnostic.py $P
for d in main_seed0 main_seed1 main_seed2 chrom; do
  nice -n 10 $PY -I $P/scripts/evaluate.py $P $W/cand3_${d}_predictions.tsv.gz $W/stack_${d}_predictions.tsv.gz \
    --matched --n 1000 --out $P/results/eval_v1_cand3_vs_stack_nosim_${d}.tsv > /dev/null &
done
wait
$PY -I $P/scripts/similarity_diagnostic.py $P --summary
echo "end $(date)"
