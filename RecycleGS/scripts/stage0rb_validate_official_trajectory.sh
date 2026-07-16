#!/usr/bin/env bash
set -euo pipefail

cd /data/wyh/RecycleGS

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS:/data/wyh/repos/TSGS:${PYTHONPATH:-}

MODEL=/data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k

echo "1/7 Audit training completion"
python3 src/baseline/audit_official_training_completion.py \
  --model-dir "$MODEL" \
  --log outputs/debug/stage0r/training_30k.log

echo "2/7 Audit official cfg"
python3 src/baseline/audit_official_model_cfg.py \
  --model-dir "$MODEL"

echo "3/7 Audit trajectory artifacts"
python3 src/baseline/audit_training_trajectory_artifacts.py \
  --model-dir "$MODEL" \
  --iterations 7000 10000 15000 20000 30000

echo "4/7 Verify 15k mid-run checkpoint"
python3 src/baseline/verify_midrun_checkpoint.py \
  --model-dir "$MODEL" \
  --iteration 15000

echo "5/7 Audit official stage transition"
python3 src/baseline/audit_official_stage_transition.py \
  --model-dir "$MODEL"

echo "6/7 Evaluate official trajectory and geometry"
python3 src/baseline/evaluate_official_tsgs_trajectory.py \
  --model-dir "$MODEL" \
  --iterations 7000 10000 15000 20000 30000

python3 src/baseline/evaluate_official_tsgs_geometry.py \
  --model-dir "$MODEL" \
  --iterations 15000 30000

echo "7/7 Build Stage 0R final report"
python3 src/baseline/build_stage0r_rebase_report.py

echo "Stage 0R-B validation completed."
