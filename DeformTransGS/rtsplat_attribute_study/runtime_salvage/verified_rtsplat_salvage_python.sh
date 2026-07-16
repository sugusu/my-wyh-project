#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="/data/wyh/repos/RT-Splatting:/data/wyh/repos/nvdiffrast:/home/wyh/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
echo "RT-Splatting salvage runtime is not verified: missing nvdiffrast/diff_surfel critical extensions" >&2
exec /usr/bin/python3 "$@"
