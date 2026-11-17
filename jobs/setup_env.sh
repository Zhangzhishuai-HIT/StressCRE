#!/bin/bash
# StressCRE mamba env, stage 1 (CPU stack for labels, features, baselines, motif tools).
# Stage 2 (GPU: torch 2.5.1+cu124 with mamba_ssm/causal_conv1d for PlantCAD2) is added before plan
# step 4's PlantCAD2 candidate; see the README (environment).
# Why not clone zuowu_genomemodel (2026-11-10): mamba 2.4 has no --clone; conda's clone re-downloads
# every package (its pkgs_dirs lack mamba's cache) and fails on unmirrored nvidia packages
# (anaconda.org, HTTP 000); with CONDA_PKGS_DIRS + --offline it still rejects the cache, because the
# source env records conda.anaconda.org URLs while the cached packages came from the TUNA mirror.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY   # direct; conda uses the TUNA mirror
export MAMBA_ROOT_PREFIX=/home/user/zhangzhishuai/.local/share/mamba
MAMBA=/home/user/zhangzhishuai/miniforge3/bin/mamba
E=$MAMBA_ROOT_PREFIX/envs/stresscre
echo "start $(date)"
$MAMBA create -y -p $E --override-channels -c conda-forge -c bioconda \
    python=3.10 numpy pandas scipy scikit-learn lightgbm matplotlib pyarrow openpyxl h5py \
    pyfaidx bedtools blast meme logomaker
echo "created $(date)"
$E/bin/python -m pip install --no-input modisco-lite
# added later with pip (variant tests, W05 alignments, Koh et al. adapters): statsmodels 0.14.5, edlib 1.3.9, peft 0.18.1
$E/bin/python -m pip install --no-input statsmodels==0.14.5 edlib==1.3.9.post1 peft==0.18.1
echo "pip done $(date)"
$E/bin/python -c "import numpy, pandas, sklearn, lightgbm, pyfaidx, modiscolite; print('numpy', numpy.__version__, '| pandas', pandas.__version__, '| sklearn', sklearn.__version__, '| lightgbm', lightgbm.__version__)"
for t in blastn makeblastdb bedtools tomtom fimo streme; do printf "%s: " $t; [ -x $E/bin/$t ] && echo ok || echo MISSING; done
echo "done $(date)"
