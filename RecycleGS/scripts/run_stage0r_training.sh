#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

cd /data/wyh/repos/TSGS
rm -rf /data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k

exec python3 train.py \
  -s /data/wyh/RecycleGS/data/translab_full/scene_01 \
  -m /data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k \
  --delight --normal --eval --use_asg --resolution 2 \
  --iterations 30000 --sd_normal_until_iter 30000 \
  --delight_iterations 15000 --normal_cos_threshold_iter 3000 \
  --ncc_loss_from_iter 7000 \
  --nofix_position --nofix_scaling --nofix_rotation \
  --seed 42 \
  --test_iterations 7000 10000 15000 20000 30000 \
  --save_iterations 7000 10000 15000 20000 30000 \
  --checkpoint_iterations 7000 10000 15000 20000
