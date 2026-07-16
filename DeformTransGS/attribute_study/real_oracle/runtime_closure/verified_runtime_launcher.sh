#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONPATH="/data/wyh/repos/TSGS/submodules/diff-first-surface-rasterization/build/lib.linux-x86_64-cpython-310:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
exec /usr/bin/python3 "$@"
