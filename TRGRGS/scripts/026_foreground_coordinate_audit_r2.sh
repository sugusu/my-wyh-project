#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wyh/TRGRGS;PY=/data/wyh/RecycleGS_Mainline/envs/tsgs_mainline/bin/python;export CUDA_VISIBLE_DEVICES=2;export PYTHONPATH="$ROOT/compat:$ROOT/third_party/TSGS:${PYTHONPATH:-}"
mkdir -p "$ROOT/outputs/scene_01/foreground_audit_r2"
echo "CUDA_VISIBLE_DEVICES=2 $PY $ROOT/tools/extract_tsgs_gaussians_r2.py" > "$ROOT/logs/stage06_r2_commands.txt"
echo "CUDA_VISIBLE_DEVICES=2 $PY $ROOT/tools/run_foreground_audit_r2.py" >> "$ROOT/logs/stage06_r2_commands.txt"
$PY "$ROOT/tools/extract_tsgs_gaussians_r2.py" 2>&1 | tee "$ROOT/logs/stage06_r2_extract.log"
$PY "$ROOT/tools/run_foreground_audit_r2.py" 2>&1 | tee "$ROOT/logs/stage06_r2_audit.log"
