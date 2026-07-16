#!/usr/bin/env bash
set -euo pipefail
# Stage 2B-Z: Final Recovery Verification
# Unified evaluator + schedule_control 500-step + random/mask_risk + Gate 2B

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH="${PROJECT_DIR}/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=7.0

PREFIX="[Stage 2B-Z]"
echo "$PREFIX Starting Stage 2B-Z: Final Recovery Verification"

# Step 1: Unify evaluator - verify baseline PSNR = 22.39
echo "$PREFIX Step 1: Verifying unified evaluator baseline..."
SCENE01_CONFIG="${PROJECT_DIR}/configs/stage1/reliability_scene01.yaml"
BASELINE_PLY="${PROJECT_DIR}/baselines/tsgs_scene01_full/point_cloud/iteration_15000/point_cloud.ply"
OUT_DIR="${PROJECT_DIR}/outputs/debug/stage2b_eval"
mkdir -p "$OUT_DIR"

python3 "${PROJECT_DIR}/src/evaluation/unified_recovery_evaluator.py" \
  --scene-config "$SCENE01_CONFIG" \
  --ply-path "$BASELINE_PLY" \
  --output "${OUT_DIR}/unified_baseline_scene01.json" \
  --tag "baseline-scene01"

# Step 2: Run schedule_control 500-step recovery with --no-delight
echo "$PREFIX Step 2: Running schedule_control 500-step recovery..."
rm -rf "${PROJECT_DIR}/outputs/recovery/scene_01/schedule_control"
python3 "${PROJECT_DIR}/src/prune/train_pruned_recovery.py" \
  --scene-config "$SCENE01_CONFIG" \
  --recovery-config "${PROJECT_DIR}/configs/stage2/recovery_500_locked.yaml" \
  --method schedule_control \
  --output-dir "${PROJECT_DIR}/outputs/recovery/scene_01/schedule_control" \
  --seed 0 \
  --no-delight

# Evaluate schedule_control recovery
echo "$PREFIX Evaluating schedule_control recovery..."
python3 "${PROJECT_DIR}/src/evaluation/unified_recovery_evaluator.py" \
  --scene-config "$SCENE01_CONFIG" \
  --ply-path "${PROJECT_DIR}/outputs/recovery/scene_01/schedule_control/point_cloud/iteration_15500/point_cloud.ply" \
  --output "${OUT_DIR}/scene01_sc500_final.json" \
  --tag "scene01-sc500"

# Step 3: Run random recovery
echo "$PREFIX Step 3: Running random 500-step recovery..."
rm -rf "${PROJECT_DIR}/outputs/recovery/scene_01/random"
python3 "${PROJECT_DIR}/src/prune/train_pruned_recovery.py" \
  --scene-config "$SCENE01_CONFIG" \
  --recovery-config "${PROJECT_DIR}/configs/stage2/recovery_500_locked.yaml" \
  --method random \
  --output-dir "${PROJECT_DIR}/outputs/recovery/scene_01/random" \
  --removed-indices "${PROJECT_DIR}/outputs/prune_only/scene_01/ratio_005/random/prune_indices.npy" \
  --seed 0 \
  --no-delight

python3 "${PROJECT_DIR}/src/evaluation/unified_recovery_evaluator.py" \
  --scene-config "$SCENE01_CONFIG" \
  --ply-path "${PROJECT_DIR}/outputs/recovery/scene_01/random/point_cloud/iteration_15500/point_cloud.ply" \
  --output "${OUT_DIR}/scene01_random500_final.json" \
  --tag "scene01-random500"

# Run mask_risk recovery
echo "$PREFIX Running mask_risk 500-step recovery..."
rm -rf "${PROJECT_DIR}/outputs/recovery/scene_01/mask_risk"
python3 "${PROJECT_DIR}/src/prune/train_pruned_recovery.py" \
  --scene-config "$SCENE01_CONFIG" \
  --recovery-config "${PROJECT_DIR}/configs/stage2/recovery_500_locked.yaml" \
  --method mask_risk \
  --output-dir "${PROJECT_DIR}/outputs/recovery/scene_01/mask_risk" \
  --removed-indices "${PROJECT_DIR}/outputs/prune_only/scene_01/ratio_005/mask_risk/prune_indices.npy" \
  --seed 0 \
  --no-delight

python3 "${PROJECT_DIR}/src/evaluation/unified_recovery_evaluator.py" \
  --scene-config "$SCENE01_CONFIG" \
  --ply-path "${PROJECT_DIR}/outputs/recovery/scene_01/mask_risk/point_cloud/iteration_15500/point_cloud.ply" \
  --output "${OUT_DIR}/scene01_maskrisk500_final.json" \
  --tag "scene01-maskrisk500"

# Step 5: Re-evaluate Gate 2B
echo "$PREFIX Step 5: Running Gate 2B final evaluation..."
python3 "${PROJECT_DIR}/src/prune/evaluate_stage2b_gate_final.py"

echo "$PREFIX Done! See outputs/prune_only/stage2b_final_gate_report.md"
echo "$PREFIX Final Gate 2B verdict follows:"
grep -A1 "Verdict:" "${PROJECT_DIR}/outputs/prune_only/stage2b_final_gate_report.md" || true
