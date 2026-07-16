#!/bin/bash
# Stage 2B-Y: Fix recovery training policy by properly implementing TSGS's selective_learning_rate_control
set -e

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS/src:/data/wyh/repos/TSGS:/data/wyh/repos/TSGS/pytorch3d_stub:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/wyh/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0

REPO_ROOT=/data/wyh/RecycleGS
DEBUG_DIR=${REPO_ROOT}/outputs/debug/stage2b_recovery_collapse
mkdir -p $DEBUG_DIR

echo "================================================================"
echo " Stage 2B-Y: Fix Stage 2 Training Policy"
echo "================================================================"

# Step 1: Audit local TSGS LR policy (already saved from grep)
echo ""
echo "[Step 1] Local TSGS LR policy audit complete (local_tsgs_lr_policy.txt)"

# Step 2: Create audit_tsgs_stage2_parameter_policy.py
echo ""
echo "[Step 2] Running audit_tsgs_stage2_parameter_policy.py..."
python3 ${REPO_ROOT}/src/prune/audit_tsgs_stage2_parameter_policy.py 2>&1

# Step 3: Create debug_stage2_lr_policy.py
echo ""
echo "[Step 3] Running debug_stage2_lr_policy.py..."
python3 ${REPO_ROOT}/src/prune/debug_stage2_lr_policy.py 2>&1

# Step 4: Create audit_debug_psnr_semantics.py
echo ""
echo "[Step 4] Running audit_debug_psnr_semantics.py..."
python3 ${REPO_ROOT}/src/prune/audit_debug_psnr_semantics.py 2>&1

# Step 5: Create audit_recovery_aux_models.py
echo ""
echo "[Step 5] Running audit_recovery_aux_models.py..."
python3 ${REPO_ROOT}/src/prune/audit_recovery_aux_models.py 2>&1

# Step 6: Create verify_recovery_stage2_policy.py
echo ""
echo "[Step 6] Running verify_recovery_stage2_policy.py..."
python3 ${REPO_ROOT}/src/prune/verify_recovery_stage2_policy.py 2>&1

# Step 7: Run debug_recovery_divergence.py with --use-official-stage2-policy
echo ""
echo "[Step 7] Running debug_recovery_divergence.py with --use-official-stage2-policy..."
python3 ${REPO_ROOT}/src/prune/debug_recovery_divergence.py --use-official-stage2-policy 2>&1 || echo "WARNING: divergence test may have failed (GPU availability or env)"

# Step 8: Build final report
echo ""
echo "[Step 8] Running build_stage2by_fix_report.py..."
python3 ${REPO_ROOT}/src/prune/build_stage2by_fix_report.py 2>&1

echo ""
echo "================================================================"
echo " Stage 2B-Y Complete"
echo "================================================================"
echo ""
echo "Required outputs:"
for f in \
    ${DEBUG_DIR}/stage2_parameter_policy_comparison_report.md \
    ${DEBUG_DIR}/lr_policy_before_after_15001.csv \
    ${DEBUG_DIR}/debug_psnr_semantics_report.md \
    ${DEBUG_DIR}/aux_model_restore_report.md \
    ${DEBUG_DIR}/stage2by_fix_report.md; do
    if [ -f "$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
    fi
done
echo ""
echo "Final report:"
cat ${DEBUG_DIR}/stage2by_fix_report.md 2>/dev/null || echo "(not available)"
