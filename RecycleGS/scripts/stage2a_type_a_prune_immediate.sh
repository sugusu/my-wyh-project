#!/bin/bash
set -e

# Stage 2A: Type-A Prune-Only (Mask-Risk) with Immediate Evaluation
# Mask-risk prune-only experiment, no retraining, no reseeding.

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

cd /data/wyh/RecycleGS
STAMP=$(date '+%Y%m%d_%H%M%S')
echo "=== Stage 2A: Type-A Mask-Risk Prune-Only ==="
echo "Timestamp: $STAMP"

CONFIG_S01=configs/stage1/reliability_scene01.yaml
CONFIG_S03=configs/stage1/reliability_scene03.yaml
LOCKED=configs/stage2/type_a_prune_locked.yaml

# ===== Step 1: Backup =====
echo ""
echo "[Step 1/9] Backup..."
mkdir -p outputs/archive
cp -r outputs/reliability outputs/archive/reliability_before_stage2a_${STAMP} 2>/dev/null || true
cp configs/stage1/gate1h_locked.yaml outputs/archive/gate1h_locked_${STAMP}.yaml 2>/dev/null || true
mkdir -p outputs/debug/stage2a_audit outputs/prune_only
echo "  Done."

# ===== Step 2: Ensure base features exist for scene_03 =====
echo ""
echo "[Step 2/9] Checking scene_03 gaussian_base_features..."
if [ ! -f "outputs/reliability/scene_03/gaussian_base_features.npz" ]; then
    echo "  Extracting gaussian_base_features for scene_03..."
    python3 src/features/extract_gaussian_base.py --config $CONFIG_S03
    echo "  Done."
else
    echo "  Already exists."
fi

# ===== Step 3: Ensure contribution exists for scene_03 =====
echo ""
echo "[Step 3/9] Checking scene_03 contribution..."
if [ ! -f "outputs/reliability/scene_03/contribution.npy" ]; then
    echo "  Extracting contribution for scene_03..."
    python3 src/features/extract_contribution.py --config $CONFIG_S03
    echo "  Done."
else
    echo "  Already exists."
fi

# ===== Step 4: Run audit =====
echo ""
echo "[Step 4/9] Running stage 1H pass audit..."
python3 src/reliability/audit_stage1h_pass.py \
    --scene01-config $CONFIG_S01 \
    --scene03-config $CONFIG_S03 \
    --locked-config $LOCKED
echo "  Done."

# ===== Step 5: Build eligible pool =====
echo ""
echo "[Step 5/9] Building eligible pool..."
SCENES_S01="scene_01"
SCENES_S03="scene_03"

for scene_config in $CONFIG_S01 $CONFIG_S03; do
    scene_name=$(grep -oP 'scene_name:\s*\K\S+' $scene_config)
    echo "  Processing $scene_name..."
    python3 src/prune/build_type_a_eligible_pool.py \
        --config $scene_config \
        --locked-config $LOCKED
    echo "  Done: $scene_name"
done

# ===== Step 6: Select prune indices =====
echo ""
echo "[Step 6/9] Selecting prune indices..."
for scene_config in $CONFIG_S01 $CONFIG_S03; do
    scene_name=$(grep -oP 'scene_name:\s*\K\S+' $scene_config)
    echo "  Processing $scene_name..."
    python3 src/prune/select_prune_indices.py \
        --config $scene_config \
        --locked-config $LOCKED \
        --methods random,low_opacity,low_contribution,mask_risk,oracle
    echo "  Done: $scene_name"
done

# ===== Step 7: Prune Gaussian PLY =====
echo ""
echo "[Step 7/9] Pruning Gaussian PLYs..."
for scene_config in $CONFIG_S01 $CONFIG_S03; do
    scene_name=$(grep -oP 'scene_name:\s*\K\S+' $scene_config)
    echo "  Processing $scene_name..."
    python3 src/prune/prune_gaussian_ply.py \
        --config $scene_config \
        --locked-config $LOCKED \
        --all-methods
    echo "  Done: $scene_name"
done

# ===== Step 8: Evaluate pruned checkpoints =====
echo ""
echo "[Step 8/9] Evaluating pruned checkpoints (this may take a while)..."
for scene_config in $CONFIG_S01 $CONFIG_S03; do
    scene_name=$(grep -oP 'scene_name:\s*\K\S+' $scene_config)
    echo "  Evaluating $scene_name..."
    python3 src/evaluation/evaluate_pruned_checkpoint.py \
        --config $scene_config \
        --locked-config $LOCKED \
        --all-methods
    echo "  Done: $scene_name"
done

# ===== Step 9: Evaluate Stage 2A Gate =====
echo ""
echo "[Step 9/9] Evaluating Stage 2A Gate..."
python3 src/prune/evaluate_stage2a_gate.py \
    --scene01 $CONFIG_S01 \
    --scene03 $CONFIG_S03 \
    --locked-config $LOCKED
echo "  Done."

# ===== Verify outputs =====
echo ""
echo "=== Verifying outputs ==="
REQUIRED=(
    "outputs/debug/stage2a_audit/scene_01_audit.json"
    "outputs/debug/stage2a_audit/scene_03_audit.json"
    "outputs/prune_only/scene_01/ratio_005/eligible_indices.npy"
    "outputs/prune_only/scene_01/ratio_005/eligible_pool_stats.json"
    "outputs/prune_only/scene_01/ratio_005/prune_metadata.json"
    "outputs/prune_only/scene_01/ratio_005/mask_risk/prune_indices.npy"
    "outputs/prune_only/scene_01/ratio_005/mask_risk/retained.ply"
    "outputs/prune_only/scene_01/ratio_005/mask_risk/removed.ply"
    "outputs/prune_only/scene_01/ratio_005/random/prune_indices.npy"
    "outputs/prune_only/scene_01/ratio_005/random/retained.ply"
    "outputs/prune_only/scene_01/ratio_005/oracle/prune_indices.npy"
    "outputs/prune_only/scene_03/ratio_005/eligible_indices.npy"
    "outputs/prune_only/scene_03/ratio_005/prune_metadata.json"
    "outputs/prune_only/stage2a_cross_scene_metrics.json"
    "outputs/prune_only/stage2a_gate_report.md"
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
echo "=== Stage 2A Complete ==="
if [ $MISSING -eq 0 ]; then
    echo "All required outputs present."
else
    echo "$MISSING required outputs missing."
fi

# Show gate report
echo ""
echo "=== Stage 2A Gate Result ==="
if [ -f "outputs/prune_only/stage2a_gate_report.md" ]; then
    head -10 outputs/prune_only/stage2a_gate_report.md
fi
echo ""
echo "Full gate report: outputs/prune_only/stage2a_gate_report.md"
