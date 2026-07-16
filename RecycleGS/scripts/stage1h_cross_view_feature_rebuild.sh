#!/bin/bash
set -e

# Stage 1H: Cross-View Reliability Feature Rebuild
# Final feature redesign for cross-view reliability check.
# If this fails, the Gaussian-level reliability approach is abandoned.

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

cd /data/wyh/RecycleGS
STAMP=$(date '+%Y%m%d_%H%M%S')

echo "=== Stage 1H: Cross-View Feature Rebuild ==="
echo "Timestamp: $STAMP"

CONFIG_S01=configs/stage1/reliability_scene01.yaml
CONFIG_S03=configs/stage1/reliability_scene03.yaml
LOCKED=configs/stage1/gate1h_locked.yaml

mkdir -p outputs/debug/stage1h_scene01 outputs/debug/stage1h_scene03

# Step 1: Extract mask consistency v2 (scene_01)
echo ""
echo "[Step 1/8] Extracting mask consistency v2 — scene_01..."
python3 src/features/extract_mask_consistency_v2.py \
    --config $CONFIG_S01 \
    --iteration 15000
echo "  Done."

# Step 2: Extract mask consistency v2 (scene_03)
echo ""
echo "[Step 2/8] Extracting mask consistency v2 — scene_03..."
python3 src/features/extract_mask_consistency_v2.py \
    --config $CONFIG_S03 \
    --iteration 15000
echo "  Done."

# Step 3: Extract PCA normal conflict (scene_01)
echo ""
echo "[Step 3/8] Extracting PCA normal conflict — scene_01..."
python3 src/features/extract_pca_normal_conflict.py \
    --config $CONFIG_S01 \
    --iteration 15000
echo "  Done."

# Step 4: Extract PCA normal conflict (scene_03)
echo ""
echo "[Step 4/8] Extracting PCA normal conflict — scene_03..."
python3 src/features/extract_pca_normal_conflict.py \
    --config $CONFIG_S03 \
    --iteration 15000
echo "  Done."

# Step 5: Evaluate stage 1H features (cross-scene)
echo ""
echo "[Step 5/8] Evaluating stage 1H features..."
python3 src/reliability/evaluate_stage1h_features.py \
    --scene01-config $CONFIG_S01 \
    --scene03-config $CONFIG_S03 \
    --locked-config $LOCKED
echo "  Done."

# Step 6: Evaluate cross-scene gate 1H
echo ""
echo "[Step 6/8] Evaluating cross-scene gate 1H..."
python3 src/reliability/evaluate_cross_scene_gate1h.py
echo "  Done."

# Step 7: Verify outputs
echo ""
echo "[Step 7/8] Verifying required outputs..."
REQUIRED=(
    "outputs/reliability/scene_01/iter_15000/mask_consistency_stats.json"
    "outputs/reliability/scene_01/iter_15000/mask_risk_cv.npy"
    "outputs/reliability/scene_01/iter_15000/pca_normal_stats.json"
    "outputs/reliability/scene_01/iter_15000/pca_normal_conflict.npy"
    "outputs/reliability/scene_03/iter_15000/mask_consistency_stats.json"
    "outputs/reliability/scene_03/iter_15000/mask_risk_cv.npy"
    "outputs/reliability/scene_03/iter_15000/pca_normal_stats.json"
    "outputs/reliability/scene_03/iter_15000/pca_normal_conflict.npy"
    "outputs/reliability/stage1h_cross_scene_feature_metrics.json"
    "outputs/reliability/stage1h_cross_scene_feature_report.md"
    "outputs/reliability/stage1h_final_gate.json"
    "outputs/reliability/stage1h_final_gate.md"
)
MISSING=0
for f in "${REQUIRED[@]}"; do
    if [ -f "$f" ]; then
        echo "  OK   $f"
    else
        echo "  MISSING: $f"
        MISSING=$((MISSING+1))
    fi
done

echo ""
echo "=== Stage 1H Complete ==="
if [ $MISSING -eq 0 ]; then
    echo "All required outputs present."
else
    echo "$MISSING required outputs missing."
fi

# Step 8: Show final gate
echo ""
echo "[Step 8/8] Stage 1H final gate:"
if [ -f "outputs/reliability/stage1h_final_gate.md" ]; then
    head -20 outputs/reliability/stage1h_final_gate.md
fi
echo ""
echo "Final gate: outputs/reliability/stage1h_final_gate.md"
echo "Full report: outputs/reliability/stage1h_cross_scene_feature_report.md"
