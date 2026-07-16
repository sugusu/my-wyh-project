#!/bin/bash
set -e

# Stage 2A.5: Audit pruned evaluation, fix rendering, compute geometry & stability
echo "=== Stage 2A.5: Prune Evaluation Audit ==="
STAMP=$(date '+%Y%m%d_%H%M%S')
echo "Timestamp: $STAMP"

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

cd /data/wyh/RecycleGS

CONFIG_S01=configs/stage1/reliability_scene01.yaml
CONFIG_S03=configs/stage1/reliability_scene03.yaml
LOCKED=configs/stage2/type_a_prune_locked.yaml

mkdir -p outputs/debug/stage2a_audit outputs/prune_only/stage2a5

# ===== Step 1: Audit pruned model loading =====
echo ""
echo "[Step 1/6] Audit pruned model loading..."
python3 src/prune/audit_pruned_evaluation.py \
    --locked-config $LOCKED \
    --output-dir outputs/debug/stage2a_audit
echo "  Done."

# ===== Step 2: Re-evaluate pruned checkpoints with force-render =====
echo ""
echo "[Step 2/6] Re-evaluating pruned checkpoints (force-render, correct PLY loading)..."
for scene_config in $CONFIG_S01 $CONFIG_S03; do
    scene_name=$(grep -oP 'scene_name:\s*\K\S+' $scene_config)
    echo "  Evaluating $scene_name..."
    python3 src/evaluation/evaluate_pruned_checkpoint.py \
        --config $scene_config \
        --locked-config $LOCKED \
        --all-methods \
        --force-render \
        --precision 8
    echo "  Done: $scene_name"
done

# ===== Step 3: Evaluate pruned geometry =====
echo ""
echo "[Step 3/6] Evaluating pruned geometry..."
for scene_config in $CONFIG_S01 $CONFIG_S03; do
    scene_name=$(grep -oP 'scene_name:\s*\K\S+' $scene_config)
    echo "  Evaluating $scene_name..."
    python3 src/evaluation/evaluate_pruned_geometry.py \
        --config $scene_config \
        --locked-config $LOCKED \
        --all-methods
    echo "  Done: $scene_name"
done

# ===== Step 4: Evaluate random seed stability =====
echo ""
echo "[Step 4/6] Evaluating random seed stability..."
for scene_config in $CONFIG_S01 $CONFIG_S03; do
    scene_name=$(grep -oP 'scene_name:\s*\K\S+' $scene_config)
    echo "  Evaluating $scene_name..."
    python3 src/prune/evaluate_random_seed_stability.py \
        --config $scene_config \
        --locked-config $LOCKED \
        --seeds 10
    echo "  Done: $scene_name"
done

# ===== Step 5: Evaluate Stage 2A.5 gate =====
echo ""
echo "[Step 5/6] Evaluating Stage 2A.5 gate..."
python3 src/prune/evaluate_stage2a5_gate.py \
    --locked-config $LOCKED
echo "  Done."

# ===== Step 6: Verify outputs =====
echo ""
echo "[Step 6/6] Verifying outputs..."
REQUIRED=(
    "outputs/debug/stage2a_audit/pruned_model_load_audit.json"
    "outputs/debug/stage2a_audit/pruned_model_load_audit_report.md"
    "outputs/prune_only/scene_01/ratio_005/cross_method_evaluation.json"
    "outputs/prune_only/scene_03/ratio_005/cross_method_evaluation.json"
    "outputs/prune_only/scene_01/ratio_005/cross_method_geometry.json"
    "outputs/prune_only/scene_03/ratio_005/cross_method_geometry.json"
    "outputs/prune_only/scene_01/ratio_005/random_seed_stability.json"
    "outputs/prune_only/scene_03/ratio_005/random_seed_stability.json"
    "outputs/prune_only/stage2a5/stage2a5_cross_scene_metrics.json"
    "outputs/prune_only/stage2a5/stage2a5_gate_report.md"
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
echo "=== Stage 2A.5 Complete ==="
if [ $MISSING -eq 0 ]; then
    echo "All required outputs present."
else
    echo "$MISSING required outputs missing."
fi

echo ""
echo "=== Stage 2A.5 Gate Result ==="
if [ -f "outputs/prune_only/stage2a5/stage2a5_gate_report.md" ]; then
    head -6 outputs/prune_only/stage2a5/stage2a5_gate_report.md
fi
echo ""
echo "Full gate report: outputs/prune_only/stage2a5/stage2a5_gate_report.md"
