#!/bin/bash
#SBATCH --job-name=scre_var
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/variant_analysis.%j.log
# Population patterns of the predicted SNP effects (scripts/variant_analysis.py): StressCRE on the rows scored by both
# networks (when variants/delta_pc2*.npy exist) and the convolutional network alone on all rows.
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin/python
echo "start $(date) $(hostname)"
$PY -I $P/scripts/variant_analysis.py $P
echo "end $(date)"
