#!/usr/bin/env bash
set -euo pipefail

cd /data/wyh/RecycleGS

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wyh/RecycleGS:/data/wyh/repos/TSGS:${PYTHONPATH:-}

mkdir -p outputs/debug/stage0rc

echo "1/7 Audit config provenance"
python3 src/baseline/audit_training_config_provenance.py \
  --model-dir \
  /data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k \
  --log outputs/debug/stage0r/training_30k.log

echo "2/7 Resolve exact official TransLab wrapper"
python3 src/baseline/audit_exact_translab_wrapper.py

echo "3/7 Audit OOM traceback"
python3 src/baseline/audit_training_oom.py \
  --model-dir \
  /data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k \
  --log outputs/debug/stage0r/training_30k.log

echo "4/7 Analyze Gaussian growth"
python3 src/baseline/analyze_gaussian_growth.py \
  --model-dir \
  /data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k

echo "5/7 Compare failed run with exact wrapper"
python3 src/baseline/compare_failed_run_to_exact_wrapper.py \
  --model-dir \
  /data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k

echo "6/7 Audit sparse layout"
python3 src/baseline/audit_translab_sparse_layout.py \
  --scene-dir \
  /data/wyh/RecycleGS/data/translab_full/scene_01

echo "7/7 Build Stage 0R-C report"
python3 src/baseline/build_stage0rc_report.py

echo "Stage 0R-C audit completed."
