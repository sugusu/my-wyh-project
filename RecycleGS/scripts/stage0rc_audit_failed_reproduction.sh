#!/usr/bin/env bash
set -euo pipefail

cd /data/wyh/RecycleGS

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS:/data/wyh/repos/TSGS:${PYTHONPATH:-}

mkdir -p outputs/debug/stage0rc

FAILED_MODEL=/data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k
FAILED_LOG=outputs/debug/stage0r/training_30k.log
FALLBACK_MODEL=/data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k_v4

echo "0/8 Audit deleted Stage 0R evidence"
python3 src/baseline/audit_deleted_stage0rc_evidence.py

if [[ ! -f "$FAILED_MODEL/cfg_args" && -f "$FALLBACK_MODEL/cfg_args" ]]; then
  AUDIT_MODEL="$FALLBACK_MODEL"
  AUDIT_LOG="$FALLBACK_MODEL/training_log.log"
else
  AUDIT_MODEL="$FAILED_MODEL"
  AUDIT_LOG="$FAILED_LOG"
fi

echo "1/8 Audit config provenance"
python3 src/baseline/audit_training_config_provenance.py \
  --model-dir \
  "$AUDIT_MODEL" \
  --log "$AUDIT_LOG"

echo "2/8 Resolve exact official TransLab wrapper"
python3 src/baseline/audit_exact_translab_wrapper.py

echo "3/8 Audit OOM traceback"
python3 src/baseline/audit_training_oom.py \
  --model-dir \
  "$FAILED_MODEL" \
  --log "$FAILED_LOG"

echo "4/8 Analyze Gaussian growth"
python3 src/baseline/analyze_gaussian_growth.py \
  --model-dir \
  "$AUDIT_MODEL"

echo "5/8 Compare failed run with exact wrapper"
python3 src/baseline/compare_failed_run_to_exact_wrapper.py \
  --model-dir \
  "$AUDIT_MODEL" \
  --log "$AUDIT_LOG"

echo "6/8 Audit sparse layout"
python3 src/baseline/audit_translab_sparse_layout.py \
  --scene-dir \
  /data/wyh/RecycleGS/data/translab_full/scene_01

echo "7/8 Build Stage 0R-C report"
python3 src/baseline/build_stage0rc_report.py

echo "Stage 0R-C audit completed."
