#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:${PATH}"
export TORCH_EXTENSIONS_DIR="/data/wyh/DeformTransGS/runtime/rtsplat_stage5_R2_build/torch_extensions"
export LD_LIBRARY_PATH="/home/wyh/.local/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="/home/wyh/.conda/envs/rtsplat_legacy2/include/python3.10:${CPATH:-}"
export PYTHONPATH="/data/wyh/repos/RT-Splatting:/data/wyh/DeformTransGS/runtime/rtsplat_stage5_R2_build/nvdiffrast/lib.linux-x86_64-cpython-310:/data/wyh/DeformTransGS/runtime/rtsplat_stage5_R2_build/nvdiffrast:/data/wyh/DeformTransGS/runtime/rtsplat_stage5_R2_build/diff_surfel_anych/lib.linux-x86_64-cpython-310:/data/wyh/DeformTransGS/runtime/rtsplat_stage5_R2_build/diff_surfel_anych:/data/wyh/repos/nvdiffrast:/home/wyh/.local/lib/python3.10/site-packages:/home/wyh/.conda/envs/rtsplat_legacy2/lib/python3.10/site-packages:${PYTHONPATH:-}"
exec /usr/bin/python3 "$@"
