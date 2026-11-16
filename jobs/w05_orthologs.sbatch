#!/bin/bash
#SBATCH --job-name=scre_w05o
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/w05_orthologs.%j.log
# W05 (G. soja) - Wm82.a4 one-to-one orthologs: DIAMOND reciprocal best protein hits (primary isoforms).
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
D=/home/user/zhangzhishuai/.local/share/mamba/envs/bio3/bin/diamond
L=/home/user/zhangzhishuai/myhermes/down_data/LIS_genomes_annotations
W=$P/work/w05
echo "start $(date) $(hostname)"
zcat $L/Glycine_soja/W05.gnm1.ann1.T47J/glyso.W05.gnm1.ann1.T47J.protein_primary.faa.gz > $W/w05.faa
zcat $L/Glycine_max/Wm82.gnm4.ann1.T8TQ/glyma.Wm82.gnm4.ann1.T8TQ.protein_primary.faa.gz > $W/a4.faa
$D makedb --in $W/w05.faa -d $W/w05 --quiet
$D makedb --in $W/a4.faa -d $W/a4 --quiet
$D blastp -q $W/w05.faa -d $W/a4 -o $W/w05_vs_a4.m8 --max-target-seqs 1 --evalue 1e-10 -p 8 --quiet \
   --outfmt 6 qseqid sseqid pident length qlen slen evalue bitscore
$D blastp -q $W/a4.faa -d $W/w05 -o $W/a4_vs_w05.m8 --max-target-seqs 1 --evalue 1e-10 -p 8 --quiet \
   --outfmt 6 qseqid sseqid pident length qlen slen evalue bitscore
wc -l $W/*.m8
echo "end $(date)"
