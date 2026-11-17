#!/bin/bash
#SBATCH --job-name=scre_w05a
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/w05_analysis.%j.log
# W05 salt-response validation (scripts/w05_analysis.py) -> results/w05_validation.tsv
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin/python -I $P/scripts/w05_analysis.py $P
