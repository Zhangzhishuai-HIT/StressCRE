#!/bin/bash
#SBATCH --job-name=scre_mtf
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/modisco_motifs.%j.log
# Trim the tfmodisco-lite patterns of every task, match them to JASPAR 2024 and the soybean DAP-seq library (Tomtom)
# and write results/modisco_patterns.tsv, results/modisco_seqlets.tsv.gz and work/modisco/patterns.meme.
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin/python
echo "start $(date) $(hostname)"
$PY -I $P/scripts/modisco_motifs.py $P
echo "end $(date)"
