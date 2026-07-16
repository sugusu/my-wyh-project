#!/bin/bash
set -e

# Stage 1F: Fix surface support implementation bug, rebuild all features independently for 7k and 15k
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

cd /data/wyh/RecycleGS
STAMP=$(date '+%Y%m%d_%H%M%S')

echo "=== Stage 1F: Repair Support & Signed Gate ==="
echo "Timestamp: $STAMP"

# Step 1: Backup
echo ""
echo "[Step 1/11] Backing up previous outputs..."
mkdir -p outputs/archive
cp -r outputs/debug/stage1e_scene01 outputs/archive/stage1e_invalid_conclusion_${STAMP} 2>/dev/null || true
cp -r outputs/reliability/scene_01 outputs/archive/scene01_before_stage1f_${STAMP} 2>/dev/null || true
mkdir -p outputs/debug/stage1f_scene01 outputs/reliability/scene_01/iter_7000 outputs/reliability/scene_01/iter_15000 outputs/figures/scene_01/stage1f
echo "  Done."

# Step 2: Audit feature indices
echo ""
echo "[Step 2/11] Auditing feature index consistency..."
python3 src/reliability/audit_feature_indices.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 3: Build checkpoint-specific candidate domains (7k AND 15k)
echo ""
echo "[Step 3/11] Building checkpoint candidate domains..."
python3 src/reliability/build_checkpoint_candidate_domain.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 4: Compute geometry errors for each checkpoint
echo ""
echo "[Step 4/11] Computing checkpoint geometry errors..."
python3 src/evaluation/compute_checkpoint_geometry_errors.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 5: Extract surface support v2 (kNN on candidate domain)
echo ""
echo "[Step 5/11] Extracting surface support v2 for 15k..."
python3 src/features/extract_surface_support_v2.py \
    --config configs/stage1/reliability_scene01.yaml \
    --iteration 15000
echo "  Done."

echo ""
echo "[Step 5b/11] Extracting surface support v2 for 7k..."
python3 src/features/extract_surface_support_v2.py \
    --config configs/stage1/reliability_scene01.yaml \
    --iteration 7000
echo "  Done."

# Step 6: Extract scale anomaly v2
echo ""
echo "[Step 6/11] Extracting scale anomaly v2 for 15k..."
python3 src/features/extract_scale_anomaly_v2.py \
    --config configs/stage1/reliability_scene01.yaml \
    --iteration 15000
echo "  Done."

echo ""
echo "[Step 6b/11] Extracting scale anomaly v2 for 7k..."
python3 src/features/extract_scale_anomaly_v2.py \
    --config configs/stage1/reliability_scene01.yaml \
    --iteration 7000
echo "  Done."

# Step 7: Evaluate normal validity on each checkpoint
echo ""
echo "[Step 7/11] Evaluating normal valid subset for 15k..."
python3 src/features/evaluate_normal_valid_subset.py \
    --config configs/stage1/reliability_scene01.yaml \
    --iteration 15000
echo "  Done."

echo ""
echo "[Step 7b/11] Evaluating normal valid subset for 7k..."
python3 src/features/evaluate_normal_valid_subset.py \
    --config configs/stage1/reliability_scene01.yaml \
    --iteration 7000
echo "  Done."

# Step 8: Evaluate signed continuous features
echo ""
echo "[Step 8/11] Evaluating signed continuous features for 15k..."
python3 src/reliability/evaluate_signed_continuous_features.py \
    --config configs/stage1/reliability_scene01.yaml \
    --iteration 15000
echo "  Done."

echo ""
echo "[Step 8b/11] Evaluating signed continuous features for 7k..."
python3 src/reliability/evaluate_signed_continuous_features.py \
    --config configs/stage1/reliability_scene01.yaml \
    --iteration 7000
echo "  Done."

# Step 9: Compare signed checkpoint features
echo ""
echo "[Step 9/11] Comparing signed checkpoint features..."
python3 src/reliability/compare_signed_checkpoint_features.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 10: Build Stage 1F report
echo ""
echo "[Step 10/11] Building Stage 1F report..."
python3 src/reliability/build_stage1f_report.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 11: Verify outputs
echo ""
echo "[Step 11/11] Verifying required outputs..."
REQUIRED=(
    "outputs/debug/stage1f_scene01/feature_index_audit.json"
    "outputs/debug/stage1f_scene01/signed_checkpoint_comparison.json"
    "outputs/debug/stage1f_scene01/stage1f_report.md"
    "outputs/reliability/scene_01/iter_7000/candidate_indices.npy"
    "outputs/reliability/scene_01/iter_7000/support_risk_v2.npy"
    "outputs/reliability/scene_01/iter_7000/scale_risk_v2.npy"
    "outputs/reliability/scene_01/iter_7000/signed_feature_metrics.json"
    "outputs/reliability/scene_01/iter_15000/candidate_indices.npy"
    "outputs/reliability/scene_01/iter_15000/support_risk_v2.npy"
    "outputs/reliability/scene_01/iter_15000/scale_risk_v2.npy"
    "outputs/reliability/scene_01/iter_15000/signed_feature_metrics.json"
)
MISSING=0
for f in "${REQUIRED[@]}"; do
    if [ -f "$f" ]; then
        echo "  okay $f"
    else
        echo "  MISSING: $f"
        MISSING=$((MISSING+1))
    fi
done

echo ""
echo "=== Stage 1F Complete ==="
if [ $MISSING -eq 0 ]; then
    echo "All required outputs present."
else
    echo "$MISSING required outputs missing."
fi
echo ""
echo "Final report: outputs/debug/stage1f_scene01/stage1f_report.md"
