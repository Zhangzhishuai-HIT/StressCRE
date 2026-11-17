#!/bin/bash
# In-silico insertions as run for the paper (GPU; the project copy and the trained fold models on the GPU machine).
#   bash jobs/insertion_runs.sh <model_dir>        (model_dir: the folder with the StressCRE fold models)
# 1. main run: ten elements at eight TSS positions, reverse orientation, TTS positions and pairs
# 2. single sites at -132 bp (the position of the second site in pairs that begin with an ABRE)
# 3. merge the two runs and add the change in log odds
set -euo pipefail
P=/home/user/zhangzhishuai/myhermes/2080/StressCRE
PY=${PY:-python}
M=$1
E=$P/features/insertion_elements.tsv                                  # in the code repository: config/insertion_elements.tsv
[ -f $E ] || E=$(dirname "$0")/../config/insertion_elements.tsv
$PY -I $P/scripts/insertion.py $P $E $M
$PY -I $P/scripts/insertion.py $P $E $M --offsets -132 --singles-only --out results/insertion_m132.tsv.gz
$PY -I $P/scripts/insertion_logodds.py $P --add results/insertion_m132.tsv.gz
