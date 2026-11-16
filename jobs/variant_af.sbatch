#!/bin/bash
#SBATCH --job-name=scre_vaf
#SBATCH --partition=q04,q05
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/StressCRE/logs/variant_af.%j.log
# (sample lists: info_accessions has CRLF line ends, strip \r first)
# Allele counts of the window SNPs in wild (G. soja, 203) and cultivated (G. max, 1,308) accessions of the panel VCF.
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
B=/home/user/zhangzhishuai/.local/share/mamba/envs/bio3/bin/bcftools
V=/home/user/zhangzhishuai/myhermes/down_data/LIS_glycine_max_diversity/Wm82.gnm2.div.Zhang_Jiang_2020/glyma.Wm82.gnm2.div.Zhang_Jiang_2020.SNPdata_full.vcf.gz
echo "start $(date) $(hostname)"
for g in soja max; do
  ( $B view -T $P/variants/regions.tsv -S $P/variants/samples_$g.txt $V -Ou | \
    $B query -f '%CHROM\t%POS\t%REF\t%ALT\t%INFO/AC\t%INFO/AN\n' | gzip > $P/variants/af_$g.tsv.gz ) &
done
wait
for g in soja max; do echo "$g $(zcat $P/variants/af_$g.tsv.gz | wc -l) sites"; done
echo "end $(date)"
