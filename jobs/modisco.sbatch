#!/bin/bash
#SBATCH --job-name=scre_modisco
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/modisco.%A_%a.log
# tfmodisco-lite motif discovery on the CNN contribution scores of each task's responsive genes (plan step 6).
# -n caps the seqlets per sign in input order, so the genes are shuffled first (seed 0) and the cap is the library
# default (20,000); only the two drought tasks reach it. Matching to known motifs: scripts/modisco_motifs.py.
#   sbatch jobs/modisco.sbatch      (the 2026-11-17 run was done on AutoDL with the same commands)
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
$E/python -I -c "
import numpy as np
w = '$W'
ohe, hyp = np.load(w + '/ohe.npy'), np.load(w + '/hyp.npy'); genes = open(w + '/genes.txt').read().split()
p = np.random.RandomState(0).permutation(len(genes))
np.save(w + '/ohe.npy', ohe[p]); np.save(w + '/hyp.npy', hyp[p]); open(w + '/genes.txt', 'w').write(chr(10).join(genes[i] for i in p) + chr(10))"
nice -n 10 $E/modisco motifs -s $W/ohe.npy -a $W/hyp.npy -n 20000 -w 3020 -o $W/modisco.h5 -v
echo "end $(date)"
