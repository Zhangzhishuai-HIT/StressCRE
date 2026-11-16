#!/bin/bash
#SBATCH --job-name=scre_de
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/pilot_de.%j.log
# Pilot labels for the StressCRE plan: one DESeq2 test per stress-vs-control contrast.
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
W=${W:-$P/work/de_pilot2}
R=/home/user/zhangzhishuai/.local/share/mamba/envs/soybean_r/bin/Rscript
echo "start $(date)  host $(hostname)"
nice -n 10 $R $P/scripts/de_run.R $W 8
echo "end $(date)"
