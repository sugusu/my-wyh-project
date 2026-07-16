#!/bin/bash
set -e

# Stage 2B Evaluate Only: compute PSNR/SSIM/LPIPS for recovery models and generate gate report
STAMP=$(date '+%Y%m%d_%H%M%S')
echo "=== Stage 2B Evaluate Only ==="
echo "Timestamp: $STAMP"

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

cd /data/wyh/RecycleGS

mkdir -p outputs/debug/stage2b_eval

echo ""
echo "==========================================="
echo "Step 1: Audit recovery outputs"
echo "==========================================="
python3 src/evaluation/audit_recovery_outputs.py

echo ""
echo "==========================================="
echo "Step 2: Audit metric paths"
echo "==========================================="
python3 src/evaluation/audit_stage2b_metric_paths.py

echo ""
echo "==========================================="
echo "Step 3: Evaluate recovery models: scene_01"
echo "==========================================="
for method in random low_opacity low_contribution mask_risk schedule_control; do
    echo "--- scene_01 / $method ---"
    python3 src/evaluation/evaluate_recovery_500.py --method "$method" --device cuda:0
done

echo ""
echo "==========================================="
echo "Step 4: Evaluate recovery models: scene_03"
echo "==========================================="
for method in random low_opacity low_contribution mask_risk schedule_control; do
    echo "--- scene_03 / $method ---"
    python3 src/evaluation/evaluate_recovery_500.py --method "$method" --device cuda:1
done

echo ""
echo "==========================================="
echo "Step 5: Audit recovery render load"
echo "==========================================="
python3 src/evaluation/audit_recovery_render_load.py

echo ""
echo "==========================================="
echo "Step 6: Aggregate recovery metrics"
echo "==========================================="
python3 src/evaluation/aggregate_recovery_metrics.py

echo ""
echo "==========================================="
echo "Step 7: Evaluate Stage 2B gate"
echo "==========================================="
python3 src/prune/evaluate_stage2b_gate.py

echo ""
echo "==========================================="
echo "Stage 2B Evaluate Only Complete"
echo "==========================================="
echo "Gate Report: outputs/prune_only/stage2b_gate_report_v2.md"
cat outputs/prune_only/stage2b_gate_report_v2.md
