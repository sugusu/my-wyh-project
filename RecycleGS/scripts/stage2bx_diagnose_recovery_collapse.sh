#!/bin/bash
# Stage 2B-X: Diagnose Recovery PSNR Collapse
# One-click diagnostic suite
set -e

STAMP=$(date '+%Y%m%d_%H%M%S')
echo "=== Stage 2B-X: Diagnose Recovery PSNR Collapse ==="
echo "Timestamp: $STAMP"

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

cd /data/wyh/RecycleGS

mkdir -p outputs/archive
cp -r outputs/prune_only outputs/archive/prune_only_before_recovery_debug_${STAMP} 2>/dev/null || true
mkdir -p outputs/debug/stage2b_recovery_collapse/scene_01
mkdir -p outputs/debug/stage2b_recovery_collapse/scene_03

echo ""
echo "========================================="
echo "Step 1: Debug Checkpoint Roundtrip"
echo "========================================="
python3 src/prune/debug_checkpoint_roundtrip.py 2>&1

echo ""
echo "========================================="
echo "Step 2: Compare Restored Gaussian Parameters"
echo "========================================="
python3 src/prune/compare_restored_gaussian_parameters.py 2>&1

echo ""
echo "========================================="
echo "Step 3: Audit TSGS Checkpoint Schema"
echo "========================================="
python3 src/prune/audit_tsgs_checkpoint_schema.py 2>&1

echo ""
echo "========================================="
echo "Step 4: Audit Restore Call Order"
echo "========================================="
python3 src/prune/audit_restore_call_order.py 2>&1

echo ""
echo "========================================="
echo "Step 5: Audit Recovery Iteration LR"
echo "========================================="
python3 src/prune/audit_recovery_iteration_lr.py 2>&1

echo ""
echo "========================================="
echo "Step 6: Debug Recovery Divergence (20 steps)"
echo "========================================="
python3 src/prune/debug_recovery_divergence.py 2>&1

echo ""
echo "========================================="
echo "Step 7: Native TSGS Resume Test"
echo "========================================="
bash scripts/debug_native_tsgs_resume_scene01.sh 2>&1

echo ""
echo "========================================="
echo "Step 8: Build Recovery Collapse Report"
echo "========================================="
python3 src/prune/build_recovery_collapse_report.py 2>&1

echo ""
echo "========================================="
echo "Stage 2B-X Complete"
echo "========================================="
echo ""
echo "Required outputs:"
echo "  outputs/debug/stage2b_recovery_collapse/root_cause_report.md"
echo "  outputs/debug/stage2b_recovery_collapse/scene_01/divergence_trace.csv"
echo "  outputs/debug/stage2b_recovery_collapse/scene_01/parameter_roundtrip_report.md"
echo "  outputs/debug/stage2b_recovery_collapse/checkpoint_schema_report.md"
echo "  outputs/debug/stage2b_recovery_collapse/restore_call_order_report.md"
echo "  outputs/debug/stage2b_recovery_collapse/lr_trace.csv"
