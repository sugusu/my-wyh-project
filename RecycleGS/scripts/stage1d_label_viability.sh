#!/bin/bash
set -e

# Stage 1D: Label Validity and Domain Definitions Diagnosis
# Environment
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

cd /data/wyh/RecycleGS
STAMP=$(date '+%Y%m%d_%H%M%S')

echo "=== Stage 1D: Label Viability Diagnosis ==="
echo "Timestamp: $STAMP"

# Step 1: Backup
echo ""
echo "[Step 1/9] Backing up previous outputs..."
mkdir -p outputs/archive
cp -r outputs/reliability/scene_01 outputs/archive/scene01_gate1c_failed_${STAMP} 2>/dev/null || true
cp -r outputs/debug/stage1c_scene01 outputs/archive/stage1c_failed_${STAMP} 2>/dev/null || true
mkdir -p outputs/debug/stage1d_scene01
echo "  Done."

# Step 2: Diagnose Domain GT Distribution
echo ""
echo "[Step 2/9] Diagnosing domain GT distribution..."
python3 src/evaluation/diagnose_domain_gt_distribution.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 3: Build Candidate Object Domain
echo ""
echo "[Step 3/9] Building candidate object domain..."
python3 src/reliability/build_candidate_object_domain.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 4: Compute Geometry Error Labels V2
echo ""
echo "[Step 4/9] Computing geometry error labels v2..."
python3 src/evaluation/compute_geometry_error_labels_v2.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 5: Evaluate Continuous Reliability
echo ""
echo "[Step 5/9] Evaluating continuous reliability..."
python3 src/reliability/evaluate_continuous_reliability.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 6: Compare Checkpoint Error Distribution
echo ""
echo "[Step 6/9] Comparing checkpoint error distribution..."
python3 src/evaluation/compare_checkpoint_error_distribution.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 7: Re-run object domain reliability evaluation (with improved report)
echo ""
echo "[Step 7/9] Re-running object domain reliability evaluation..."
python3 src/reliability/evaluate_object_domain_reliability.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 8: Build Stage 1D Report
echo ""
echo "[Step 8/9] Building stage 1D report..."
python3 src/reliability/build_stage1d_report.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 9: Append conclusion to realtime_log.md
echo ""
echo "[Step 9/9] Appending conclusion to realtime_log.md..."
cat >> /data/wyh/RecycleGS/realtime_log.md << 'EOF'

## Stage 1D 结论

已完成评价域和几何标签有效性诊断。

当前仍未进入 Stage 2。

下一步由 stage1d_label_viability_report.md 决定：
- A：标签恢复有效，继续设计可靠性特征；
- B：15k checkpoint 过晚，验证较早 checkpoint；
- C：单 Gaussian mesh-distance 标签不适合，暂停该检测路线。
EOF
echo "  Done."

echo ""
echo "=== Stage 1D Complete ==="
echo "Reports:"
echo "  - outputs/debug/stage1d_scene01/"
echo "  - outputs/reliability/scene_01/stage1d_label_viability_report.md"
echo "  - outputs/reliability/scene_01/stage1d_label_viability_report.json"
