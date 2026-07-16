#!/bin/bash
set -e

# Stage 2B: Recovery Experiment - 500-step retraining after mask-risk pruning
# Implements optimizer-aware prune + recovery training + evaluation

STAMP=$(date '+%Y%m%d_%H%M%S')
echo "=== Stage 2B: Recovery 500 ==="
echo "Timestamp: $STAMP"

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

cd /data/wyh/RecycleGS

mkdir -p outputs/debug/stage2b_preflight outputs/debug/stage2b_fairness

echo ""
echo "==========================================="
echo "Step 1: Verify config exists"
echo "==========================================="
ls -la configs/stage2/recovery_500_locked.yaml

echo ""
echo "==========================================="
echo "Step 2: Check checkpoint availability"
echo "==========================================="
find /data/wyh/RecycleGS/baselines/tsgs_scene01_full -maxdepth 3 -type f \( -name "*.pth" -o -name "*.pt" -o -name "*checkpoint*" -o -name "chkpnt*" \) | sort
find /data/wyh/RecycleGS/baselines/tsgs_scene03_full -maxdepth 3 -type f \( -name "*.pth" -o -name "*.pt" -o -name "*checkpoint*" -o -name "chkpnt*" \) | sort

echo ""
echo "==========================================="
echo "Step 3: Check recovery checkpoint inventory"
echo "==========================================="
python3 src/prune/check_recovery_checkpoint.py

echo ""
echo "==========================================="
echo "Step 4: Audit optimizer-aware prune"
echo "==========================================="
python3 src/prune/audit_optimizer_aware_prune.py

echo ""
echo "==========================================="
echo "Step 5: Run recovery training for all methods (scene_01)"
echo "==========================================="
python3 src/prune/run_recovery_methods.py --scene scene_01

echo ""
echo "==========================================="
echo "Step 6: Run recovery training for all methods (scene_03)"
echo "==========================================="
python3 src/prune/run_recovery_methods.py --scene scene_03

echo ""
echo "==========================================="
echo "Step 7: Evaluate recovery 500"
echo "==========================================="
python3 src/evaluation/evaluate_recovery_500.py

echo ""
echo "==========================================="
echo "Step 8: Audit recovery fairness"
echo "==========================================="
python3 src/prune/audit_recovery_fairness.py

echo ""
echo "==========================================="
echo "Step 9: Evaluate Stage 2B gate"
echo "==========================================="
python3 src/prune/evaluate_stage2b_gate.py

echo ""
echo "==========================================="
echo "Stage 2B Complete"
echo "==========================================="
echo "Gate Report: outputs/prune_only/stage2b_gate_report.md"
cat outputs/prune_only/stage2b_gate_report.md
