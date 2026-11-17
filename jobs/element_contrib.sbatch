#!/bin/bash
#SBATCH --job-name=scre_elc
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/element_contrib.%j.log
# Contribution of exact sites of known elements (scripts/abre_contrib.py).
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin/python -I $P/scripts/abre_contrib.py $P
