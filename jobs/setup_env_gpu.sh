#!/bin/bash
# StressCRE env, stage 2 (GPU): torch 2.5.1 (CUDA 12.4 build from PyPI via the Tsinghua mirror, direct)
# plus the mamba_ssm 2.2.4 / causal_conv1d 1.5.0.post8 builds that already work in zuowu_genomemodel
# (GitHub wheels cu12 + torch2.5 + cxx11abiFALSE, the same ABI as the PyPI torch 2.5.1 wheel).
# GitHub is ~1 KB/s from the cluster, so the installed package files are copied instead of re-downloaded.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
E=/home/user/zhangzhishuai/.local/share/mamba/envs/stresscre
Z=/home/user/zhangzhishuai/.local/share/mamba/envs/zuowu_genomemodel/lib/python3.10/site-packages
echo "start $(date)"
NUMPY=$($E/bin/python -c 'import numpy; print(numpy.__version__)')
$E/bin/python -m pip install --no-input -i https://pypi.tuna.tsinghua.edu.cn/simple \
    torch==2.5.1 transformers==4.46.3 einops ninja "numpy==$NUMPY" 2>&1 | grep -v "━\|Downloading\|Requirement already" | tail -8
echo "pip done $(date)"
SP=$($E/bin/python -c 'import site; print(site.getsitepackages()[0])')
for x in mamba_ssm mamba_ssm-2.2.4.dist-info causal_conv1d causal_conv1d-1.5.0.post8.dist-info \
         selective_scan_cuda.cpython-310-x86_64-linux-gnu.so causal_conv1d_cuda.cpython-310-x86_64-linux-gnu.so; do
  [ -e $SP/$x ] || cp -a $Z/$x $SP/
done
$E/bin/python -c "import torch, transformers, mamba_ssm, causal_conv1d; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| mamba_ssm', mamba_ssm.__version__, '| causal_conv1d', causal_conv1d.__version__)"
echo "done $(date)"
