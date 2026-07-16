#!/usr/bin/env bash
set -euo pipefail

# Shared Stage 0 runtime envelope. Source this file before running TSPE-GS
# commands so CUDA and network behavior are reproducible.
export CUDA_VISIBLE_DEVICES=2,3
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS="${MAX_JOBS:-4}"
export CPATH="/data/wyh/ReliablePeakGS/environment/deb_headers/extracted/usr/include/python3.10:/data/wyh/ReliablePeakGS/environment/deb_headers/extracted/usr/include:${CPATH:-}"

# The default proxy variables in this shell point at a local proxy that breaks
# GitHub. Keep them off for GitHub/PyPI/ArXiv operations unless a command
# explicitly needs Google Drive proxy routing.
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy
