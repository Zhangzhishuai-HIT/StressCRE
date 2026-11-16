#!/bin/bash
#SBATCH --job-name=scre_dap
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/dapseq_motifs.%j.log
# Soybean TF motifs from the Jiao 2024 DAP-seq peaks (148 TFs): sample peaks, STREME per TF, merge (plan step 6).
#   sbatch jobs/dapseq_motifs.sbatch     (v1: 1,000 peaks x 201 bp)
#   sbatch --export=ALL,NP=3000,WD=100,WORK=dapseq_c100,ON=soy_dapseq_streme_c100 jobs/dapseq_motifs.sbatch
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
E=/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre/bin
G=/home/user/zhangzhishuai/myhermes/down_data/LIS_genomes_annotations/Glycine_max/Wm82.gnm4.4PTR/glyma.Wm82.gnm4.4PTR.genome_main.fna.gz
D=/home/user/zhangzhishuai/myhermes/down_data/SoyTFBase_DAPseq_Jiao2024/peaks
NP=${NP:-1000}; WD=${WD:-0}; ON=${ON:-soy_dapseq_streme}
W=$P/work/${WORK:-dapseq}
echo "start $(date) $(hostname)"
mkdir -p $W/streme
$E/python -I $P/scripts/dapseq_motifs.py extract $P $G $D $W $NP $WD > $W/extract.tsv
ls $W/fa/*.fa | xargs -P 8 -I{} sh -c 'tf=$(basename {} .fa); [ -s '$W'/streme/$tf/streme.txt ] || nice -n 10 '$E'/streme --p {} --dna --nmotifs 3 --minw 6 --maxw 15 --seed 0 --verbosity 1 --oc '$W'/streme/$tf > /dev/null 2>&1'
ls $W/fa/*.fa | while read f; do tf=$(basename $f .fa); [ -d $W/streme/$tf ] || echo "missing $tf"; done
$E/python -I $P/scripts/dapseq_motifs.py merge $P $W $ON
echo "end $(date)"
