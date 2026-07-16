#!/bin/bash
# Start training jobs for Stage 2
# Model B: SH-only on GPU 2

export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib
export PYTHONPATH=/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub

rm -rf /data/wyh/DeformTransGS/baselines/tsgs_synth_canonical_sh
mkdir -p /data/wyh/DeformTransGS/baselines/tsgs_synth_canonical_sh

CUDA_VISIBLE_DEVICES=2 nohup python3 /data/wyh/repos/TSGS/train.py \
    -s /data/wyh/DeformTransGS/data/stage2_canonical_tsgs/nerf_format \
    -m /data/wyh/DeformTransGS/baselines/tsgs_synth_canonical_sh \
    --sh_degree 3 --asg_degree 24 \
    --iterations 30000 \
    --save_iterations 15000 30000 \
    --test_iterations 15000 30000 \
    --checkpoint_iterations 15000 30000 \
    --data_device cuda \
    > /data/wyh/DeformTransGS/baselines/tsgs_synth_canonical_sh/train.log 2>&1 &

echo "Model B PID: $!"

sleep 10
echo "=== Check ==="
ps aux | grep "train.py.*nerf_format" | grep -v grep
tail -3 /data/wyh/DeformTransGS/baselines/tsgs_synth_canonical_sh/train.log 2>/dev/null
