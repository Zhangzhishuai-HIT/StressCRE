#!/bin/bash
#SBATCH --job-name=scre_modisco
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/modisco.%A_%a.log
# tfmodisco-lite motif discovery on the CNN contribution scores of each task's responsive genes (plan step 6),
# then a report with Tomtom matches to JASPAR 2024 plants.   sbatch jobs/modisco.sbatch
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
E=/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin
J=/home/user/zhangzhishuai/myhermes/down_data/JASPAR2024/JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt
TASKS=(drought-up drought-down salt-up salt-down heat-up heat-down cold-up cold-down alkaline-up alkaline-down)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
W=$P/work/modisco/$T
export PATH=$E:$PATH NUMBA_NUM_THREADS=8
echo "start $(date) $(hostname) $T"
$E/python -I $P/scripts/modisco_prep.py $P/work/attr_v1 $T $W
nice -n 10 $E/modisco motifs -s $W/ohe.npy -a $W/hyp.npy -n 2000 -w 3020 -o $W/modisco.h5
nice -n 10 $E/modisco report -i $W/modisco.h5 -o $W/report -s $W/report -m $J -t
echo "end $(date)"
