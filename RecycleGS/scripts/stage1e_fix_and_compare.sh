#!/bin/bash
set -e

# Stage 1E: Fix Stage 1D report logic contradictions, diagnose feature directions, compare 7k vs 15k
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

cd /data/wyh/RecycleGS
STAMP=$(date '+%Y%m%d_%H%M%S')

echo "=== Stage 1E: Fix & Compare ==="
echo "Timestamp: $STAMP"

# Step 0: Backup
echo ""
echo "[Step 0/8] Backing up..."
mkdir -p outputs/archive
cp -r outputs/debug/stage1d_scene01 outputs/archive/stage1d_before_logic_fix_${STAMP} 2>/dev/null || true
cp -r outputs/reliability/scene_01 outputs/archive/scene01_before_stage1e_${STAMP} 2>/dev/null || true
mkdir -p outputs/debug/stage1e_scene01
echo "  Done."

# Step 1: Fix build_stage1d_report.py and run it
echo ""
echo "[Step 1/8] Running fixed stage 1D report..."
python3 src/reliability/build_stage1d_report.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 2: Fix compare_checkpoint_error_distribution.py and run it
echo ""
echo "[Step 2/8] Running fixed checkpoint comparison..."
python3 src/evaluation/compare_checkpoint_error_distribution.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 3: Diagnose surface support semantics
echo ""
echo "[Step 3/8] Diagnosing surface support semantics..."
python3 src/features/diagnose_surface_support_semantics.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 4: Diagnose scale semantics
echo ""
echo "[Step 4/8] Diagnosing scale semantics..."
python3 src/features/diagnose_scale_semantics.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 5: Diagnose normal signal validity
echo ""
echo "[Step 5/8] Diagnosing normal signal validity..."
python3 src/features/diagnose_normal_signal.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 6: Compare checkpoint reliability (7k vs 15k)
echo ""
echo "[Step 6/8] Comparing checkpoint reliability..."
python3 src/reliability/compare_checkpoint_reliability.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 7: Build stage 1E report
echo ""
echo "[Step 7/8] Building stage 1E report..."
python3 src/reliability/build_stage1e_report.py \
    --config configs/stage1/reliability_scene01.yaml
echo "  Done."

# Step 8: Verify outputs
echo ""
echo "[Step 8/8] Verifying outputs..."
REQUIRED=(
    "outputs/debug/stage1e_scene01/support_semantics.json"
    "outputs/debug/stage1e_scene01/scale_semantics.json"
    "outputs/debug/stage1e_scene01/normal_signal_validity.json"
    "outputs/debug/stage1e_scene01/checkpoint_comparison.json"
    "outputs/debug/stage1e_scene01/checkpoint_reliability_comparison.json"
    "outputs/debug/stage1e_scene01/stage1e_report.md"
)
MISSING=0
for f in "${REQUIRED[@]}"; do
    if [ -f "$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ❌ MISSING: $f"
        MISSING=$((MISSING+1))
    fi
done

echo ""
echo "=== Stage 1E Complete ==="
if [ $MISSING -eq 0 ]; then
    echo "All required outputs present."
else
    echo "$MISSING required outputs missing."
fi
echo ""
echo "Final report: outputs/debug/stage1e_scene01/stage1e_report.md"
