#!/usr/bin/env bash
set -euo pipefail

cd /data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k_v4

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k_v4/pytorch3d_stub:/data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k_v4:/data/wyh/RecycleGS:${PYTHONPATH:-}

python3 train.py -s /data/wyh/RecycleGS/data/translab_full/scene_01 -m /data/wyh/RecycleGS/baselines/tsgs_exact_scene01_30k --delight --normal --mask_background --use_asg --sd_normal_until_iter 30000 --iterations 30000 --normal_cos_threshold_iter 3000 --eval --delight_iterations 15000 --resolution 2 --ncc_loss_from_iter 7000 --nofix_position --nofix_scaling --nofix_rotation --seed 42 --normal_folder normals --use_transparencies_map --test_iterations 7000 10000 15000 20000 30000 --save_iterations 7000 10000 15000 20000 30000 --checkpoint_iterations 7000 10000 15000 20000
