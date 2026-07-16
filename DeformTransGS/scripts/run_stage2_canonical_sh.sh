#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib
export PYTHONPATH=/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub
cd /data/wyh/repos/TSGS
mkdir -p /data/wyh/DeformTransGS/baselines/tsgs_synth_canonical_sh

python3 train.py \
    -s /data/wyh/DeformTransGS/data/stage2_canonical_tsgs/dataset_full12 \
    -m /data/wyh/DeformTransGS/baselines/tsgs_synth_canonical_sh \
    --sh_degree 3 --asg_degree 24 --resolution 2 \
    --iterations 30000 \
    --save_iterations 15000 30000 \
    --test_iterations 15000 30000 \
    --checkpoint_iterations 15000 30000 \
    --data_device cuda \
    2>&1 | tee /data/wyh/DeformTransGS/baselines/tsgs_synth_canonical_sh/train.log

echo "Model B training exit: $?"
