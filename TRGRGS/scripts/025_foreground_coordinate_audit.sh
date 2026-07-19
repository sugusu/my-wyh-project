#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wyh/TRGRGS; PY=/data/wyh/RecycleGS_Mainline/envs/tsgs_mainline/bin/python
export CUDA_VISIBLE_DEVICES=2
cmd="$PY $ROOT/tools/run_foreground_audit.py"; echo "$cmd" > "$ROOT/logs/stage06_command.txt"; eval "$cmd" 2>&1 | tee "$ROOT/logs/stage06_foreground_audit.log"

