#!/bin/bash
set -e

# Stage 1G: Cross-Scene Gate 1 Validation (scene_03)
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

cd /data/wyh/RecycleGS
STAMP=$(date '+%Y%m%d_%H%M%S')
CONFIG=configs/stage1/reliability_scene03.yaml

echo "=== Stage 1G: Cross-Scene Gate 1 Validation ==="
echo "Timestamp: $STAMP"
echo "Scene: scene_03"
echo "Config: $CONFIG"

mkdir -p outputs/reliability/scene_03/iter_15000
mkdir -p outputs/debug/stage1_scene03
mkdir -p outputs/figures/scene_03

# Step 1: Validate mesh-camera-mask alignment
echo ""
echo "[Step 1/8] Validating mesh-camera-mask alignment..."
python3 src/evaluation/validate_mesh_camera_mask_alignment.py \
    --config $CONFIG
echo "  Done."

# Step 2: Build checkpoint candidate domain
echo ""
echo "[Step 2/8] Building checkpoint candidate domain..."
python3 src/reliability/build_checkpoint_candidate_domain.py \
    --config $CONFIG
echo "  Done."

# Step 3: Extract surface support v2
echo ""
echo "[Step 3/8] Extracting surface support v2 for 15k..."
python3 src/features/extract_surface_support_v2.py \
    --config $CONFIG \
    --iteration 15000
echo "  Done."

# Step 4: Extract scale anomaly v2
echo ""
echo "[Step 4/8] Extracting scale anomaly v2 for 15k..."
python3 src/features/extract_scale_anomaly_v2.py \
    --config $CONFIG \
    --iteration 15000
echo "  Done."

# Step 5: Compute checkpoint geometry errors
echo ""
echo "[Step 5/8] Computing geometry errors..."
python3 src/evaluation/compute_checkpoint_geometry_errors.py \
    --config $CONFIG
echo "  Done."

# Step 6: Evaluate signed continuous features
echo ""
echo "[Step 6/8] Evaluating signed continuous features for 15k..."
python3 src/reliability/evaluate_signed_continuous_features.py \
    --config $CONFIG \
    --iteration 15000
echo "  Done."

# Step 7: Build cross-scene Gate 1 evaluation
echo ""
echo "[Step 7/8] Evaluating cross-scene Gate 1..."
python3 src/reliability/evaluate_cross_scene_gate1.py
echo "  Done."

# Step 8: Verify outputs
echo ""
echo "[Step 8/8] Verifying outputs..."
REQUIRED=(
    "outputs/debug/stage1_scene03/mesh_camera_alignment.json"
    "outputs/debug/stage1_scene03/mesh_camera_alignment_report.md"
    "outputs/reliability/scene_03/iter_15000/candidate_indices.npy"
    "outputs/reliability/scene_03/iter_15000/support_risk_v2.npy"
    "outputs/reliability/scene_03/iter_15000/scale_risk_v2.npy"
    "outputs/reliability/scene_03/iter_15000/signed_feature_metrics.json"
    "outputs/reliability/scene_03/gate1_scene03_report.md"
    "outputs/reliability/cross_scene_gate1_metrics.json"
    "outputs/reliability/cross_scene_gate1_report.md"
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
echo "=== Stage 1G Complete ==="
if [ $MISSING -eq 0 ]; then
    echo "All required outputs present."
else
    echo "$MISSING required outputs missing."
fi
echo ""
echo "Cross-scene Gate 1 report: outputs/reliability/cross_scene_gate1_report.md"
