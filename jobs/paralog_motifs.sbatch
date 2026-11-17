#!/bin/bash
#SBATCH --job-name=scre_pmot
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/paralog_motifs.%j.log
# Known elements in the two copies of divergent WGD pairs (scripts/paralog_motifs.py) -> results/paralog_motifs.tsv
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin/python -I $P/scripts/paralog_motifs.py $P
