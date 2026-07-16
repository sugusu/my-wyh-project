#!/usr/bin/env bash
set -euo pipefail

cd /data/wyh/RecycleGS
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

CONFIG=configs/stage1/reliability_scene01.yaml

echo "1/10 坐标系检查" && python3 src/evaluation/check_coordinate_consistency.py --config $CONFIG
echo "2/10 导出冻结参考状态" && python3 src/evaluation/export_frozen_reference.py --config $CONFIG
echo "3/10 提取 Gaussian 基础属性" && python3 src/features/extract_gaussian_base.py --config $CONFIG
echo "4/10 提取 mask risk" && python3 src/features/extract_mask_support.py --config $CONFIG
echo "5/10 提取 normal conflict" && python3 src/features/extract_normal_conflict.py --config $CONFIG
echo "6/10 提取 depth-order conflict" && python3 src/features/extract_depth_order_conflict.py --config $CONFIG
echo "7/10 提取 surface support + scale" && python3 src/features/extract_surface_support.py --config $CONFIG && python3 src/features/extract_scale_anomaly.py --config $CONFIG
echo "8/10 提取 contribution" && python3 src/features/extract_contribution.py --config $CONFIG
echo "9/10 计算 GT 几何误差 + 风险" && python3 src/evaluation/compute_gaussian_gt_error.py --config $CONFIG && python3 src/reliability/compute_reliability.py --config $CONFIG
echo "10/10 Gate 1A 评价" && python3 src/reliability/evaluate_reliability.py --config $CONFIG
echo "Stage 1A scene_01 完成。禁止自动进入 prune。"
