#!/bin/bash
#SBATCH --job-name=scre_psim
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/promoter_sim.%j.log
# All-vs-all megablast of the 3,020-bp windows (CDS already masked to N; DUST on): window pairs with
# >= 80% identity over >= 200 bp become "promoter-similar" edges for the split groups (plan step 3).
# (MMseqs2 nucleotide search needed > 7 GB to index 160 Mb; megablast runs in < 2 GB.)
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
B=/home/user/zhangzhishuai/.local/share/mamba/envs/bio3/bin
W=$P/work/psim; mkdir -p $W; T=/tmp/scre_psim_$SLURM_JOB_ID; mkdir -p $T
echo "start $(date) $(hostname)"
zcat $P/seqs/wm82a4_windows.fa.gz > $T/win.fa
$B/makeblastdb -in $T/win.fa -dbtype nucl -out $T/win > /dev/null
$B/blastn -task megablast -query $T/win.fa -db $T/win -perc_identity 80 -evalue 1e-10 \
    -max_target_seqs 200 -num_threads 8 \
    -outfmt "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore" > $W/a4_win_self.m8
awk '$1 != $2 && $4 >= 200' $W/a4_win_self.m8 > $W/a4_win_edges.m8
echo "hits $(wc -l < $W/a4_win_self.m8), edges (>=200 bp, non-self) $(wc -l < $W/a4_win_edges.m8)"
rm -rf $T
echo "end $(date)"
