#!/bin/bash
#SBATCH --job-name=scre_mscan
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --array=0-22
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/motif_scan.%A_%a.log
# FIMO scan of the 805 JASPAR 2024 plant motifs over the a4 windows, 23 subsets of 35 motifs (motif baseline).
#   zcat seqs/wm82a4_windows.fa.gz > work/motif/wm82a4_windows.fa; sbatch jobs/motif_scan.sbatch
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
PY=/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin/python
M=/home/user/zhangzhishuai/myhermes/down_data/JASPAR2024/JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt
echo "start $(date) $(hostname) subset $SLURM_ARRAY_TASK_ID"
nice -n 10 $PY -I $P/scripts/motif_scan.py scan $P $P/work/motif/wm82a4_windows.fa $M $SLURM_ARRAY_TASK_ID 23 $P/work/motif
echo "end $(date)"
