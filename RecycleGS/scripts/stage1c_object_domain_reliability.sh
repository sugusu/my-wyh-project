#!/usr/bin/env bash
set -euo pipefail

cd /data/wyh/RecycleGS
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

CONFIG=configs/stage1/reliability_scene01.yaml
STAMP=$(date '+%Y%m%d_%H%M%S')

echo "=== Stage 1C: Object Domain Partition & Reliability Pipeline ==="
echo "Config: $CONFIG"
echo "Timestamp: $STAMP"
echo ""

echo "Step 0/13: Backup"
mkdir -p outputs/archive
cp -r outputs/reliability/scene_01 outputs/archive/scene01_before_object_partition_${STAMP} 2>/dev/null || true
cp -r outputs/debug/stage1b_scene01 outputs/archive/stage1b_before_object_partition_${STAMP} 2>/dev/null || true
mkdir -p outputs/debug/stage1c_scene01
echo "  Backup done."

echo ""
echo "Step 1/13: Validate mesh-camera mask alignment"
python3 src/evaluation/validate_mesh_camera_mask_alignment.py --config $CONFIG
echo "---"

echo ""
echo "Step 2/13: Extract object-domain mask support"
python3 src/features/extract_object_domain_support.py --config $CONFIG
echo "---"

echo ""
echo "Step 3/13: Partition Gaussian domains"
python3 src/reliability/partition_gaussian_domains.py --config $CONFIG
echo "---"

echo ""
echo "Step 4/13: Check object domain alignment"
python3 src/evaluation/check_object_domain_alignment.py --config $CONFIG
echo "---"

echo ""
echo "Step 5/13: Compute object-domain GT error"
python3 src/evaluation/compute_object_domain_gt_error.py --config $CONFIG
echo "---"

echo ""
echo "Step 6/13: Extract normal conflict with object domain support"
python3 src/features/extract_normal_conflict.py --config $CONFIG
echo "---"

echo ""
echo "Step 7/13: Extract object surface support"
python3 src/features/extract_object_surface_support.py --config $CONFIG
echo "---"

echo ""
echo "Step 8/13: Extract object scale anomaly"
python3 src/features/extract_object_scale_anomaly.py --config $CONFIG
echo "---"

echo ""
echo "Step 9/13: Create depth feature disabled note"
echo "See outputs/debug/stage1c_scene01/depth_feature_disabled.md"
echo "---"

echo ""
echo "Step 10/13: Compute object domain risk"
python3 src/reliability/compute_object_domain_risk.py --config $CONFIG
echo "---"

echo ""
echo "Step 11/13: Evaluate object domain reliability"
python3 src/reliability/evaluate_object_domain_reliability.py --config $CONFIG
echo "---"

echo ""
echo "Step 12/13: Append conclusion to realtime_log.md"
cat >> realtime_log.md << 'LOGEOF'

## Stage 1C 结论

已将整场景 Gaussian 划分为 object-supported、background-supported 和 uncertain。

GT mesh 只用于 object-supported Gaussian 的离线可靠性评价。

当前仍未进入 Stage 2。

下一步根据 gate1c_report.md：
- 若 scene_01 物体域初步通过：用相同参数复现 scene_03；
- 若仍失败：暂停 prune，重新评估单 Gaussian 可靠性标签和特征定义。
LOGEOF
echo "  Log appended."

echo ""
echo "Step 13/13: Pipeline complete"
echo "=== Stage 1C Pipeline Complete ==="
echo ""
echo "Key outputs:"
echo "  - outputs/reliability/scene_01/domain_colored.ply"
echo "  - outputs/reliability/scene_01/object_indices.npy"
echo "  - outputs/reliability/scene_01/object_risk_A.npy"
echo "  - outputs/reliability/scene_01/object_risk_B.npy"
echo "  - outputs/reliability/scene_01/object_risk_C.npy"
echo "  - outputs/debug/stage1c_scene01/gate1c_report.md"
