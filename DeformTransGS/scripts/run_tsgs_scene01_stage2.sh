#!/bin/bash
set -euo pipefail

# ============================================================
# run_tsgs_scene01_stage2.sh
# 从 tsgs_scene01_full Stage 1 checkpoint 继续训练 Stage 2
#
# 数据路径：/data/wyh/RecycleGS/data/translab_full/scene_01
# 源 checkpoint：/data/wyh/RecycleGS/baselines/tsgs_scene01_full/chkpnt15000.pth
# 输出目录：/data/wyh/DeformTransGS/baselines/tsgs_scene01_full_stage2
# ============================================================

export CUDA_VISIBLE_DEVICES=2,3
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:$PYTHONPATH

TSGS_DIR=/data/wyh/repos/TSGS
DATA_DIR=/data/wyh/RecycleGS/data/translab_full/scene_01
SOURCE_CKPT=/data/wyh/RecycleGS/baselines/tsgs_scene01_full/chkpnt15000.pth
OUTPUT_DIR=/data/wyh/DeformTransGS/baselines/tsgs_scene01_full_stage2

mkdir -p "$OUTPUT_DIR"

cd "$TSGS_DIR"

python3 train.py \
    -s "$DATA_DIR" \
    -m "$OUTPUT_DIR" \
    --start_checkpoint "$SOURCE_CKPT" \
    --iterations 30000 \
    --save_iterations 15000 30000 \
    --test_iterations 15000 30000 \
    --checkpoint_iterations 15000 30000 \
    --sh_degree 3 \
    --asg_degree 24 \
    --resolution 2 \
    --data_device cuda

# 说明：
# 1. --start_checkpoint 从 chkpnt15000.pth 恢复完整优化器状态
# 2. --iterations 30000 继续训练到 30000 总迭代（从 15000 开始额外 15000 步）
# 3. 使用默认 asg_degree=24（SpecularNetwork，适用于合成数据模式）
# 4. 如需真实场景模式（SpecularNetworkReal），添加 --is_real
# 5. Stage 2 中 geometry 参数默认冻结（通过 selective_learning_rate_control）
#    仅 transparency, f_dc, f_rest, f_asg 继续优化
# 6. 当前 baseline 训练时未使用 delight/normal，此处保持一致
