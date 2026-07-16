#!/usr/bin/env bash
set -euo pipefail

cd /data/wyh/RecycleGS
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

CONFIG=configs/stage1/diagnose_scene01.yaml
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=== Stage 1B Gate 1 Failure Diagnosis Pipeline ==="
echo "Config: $CONFIG"
echo "Timestamp: $TIMESTAMP"
echo ""

echo "1/10 坐标一致性检查"
python3 src/evaluation/check_coordinate_consistency.py --config $CONFIG || echo "[WARNING] Coordinate check triggered stop conditions (expected for diagnosis)"
echo "---"

echo "2/10 坐标对齐诊断"
python3 src/evaluation/diagnose_coordinate_alignment.py --config $CONFIG
echo "---"

echo "3/10 GT标签诊断"
python3 src/evaluation/diagnose_gt_labels.py --config $CONFIG
echo "---"

echo "4/10 特征诊断"
python3 src/reliability/diagnose_features.py --config $CONFIG
echo "---"

echo "5/10 掩码对齐诊断"
python3 src/features/diagnose_mask_alignment.py --config $CONFIG
echo "---"

echo "6/10 法线约定诊断"
python3 src/features/diagnose_normal_convention.py --config $CONFIG
echo "---"

echo "7/10 深度顺序诊断"
python3 src/features/diagnose_depth_order.py --config $CONFIG
echo "---"

echo "8/10 分类型可靠性评估"
python3 src/reliability/evaluate_typed_reliability.py --config $CONFIG
echo "---"

echo "9/10 构建Gate 1综合诊断"
python3 src/reliability/build_gate1_diagnosis.py --config $CONFIG
echo "---"

echo "10/10 验证输出文件"
STAGE1B_DIR=/data/wyh/RecycleGS/outputs/debug/stage1b_scene01

echo "Checking stage1_scene01 output files..."
for f in coordinate_consistency.json coordinate_check_report.md cameras_gaussians_mesh_bbox.ply; do
    if [ -f "$DEBUG_DIR/$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
    fi
done

echo "Checking stage1b_scene01 output files..."
FILES=(
    coordinate_alignment_diagnosis.json
    coordinate_alignment_report.md
    gt_label_diagnosis.json
    gt_label_report.md
    gt_label_histogram.png
    feature_diagnosis.json
    feature_diagnosis_report.md
    feature_correlation.csv
    feature_correlation.png
    normal_convention_diagnosis.json
    normal_convention_report.md
    depth_order_diagnosis.json
    depth_order_report.md
    mask_alignment_report.json
    typed_reliability_metrics.json
    typed_reliability_report.md
    gate1_failure_diagnosis.json
    gate1_failure_diagnosis.md
)
for f in "${FILES[@]}"; do
    if [ -f "$STAGE1B_DIR/$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "=== Stage 1B Pipeline Complete ==="
echo "gate1_failure_diagnosis.md content:"
echo "----------------------------------------"
cat "$STAGE1B_DIR/gate1_failure_diagnosis.md" 2>/dev/null || echo "  Not found."
echo "----------------------------------------"
