#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wyh/TRGRGS
CMD="/data/wyh/RecycleGS_Mainline/envs/tsgs_mainline/bin/python $ROOT/tools/finalize_stage15.py"
printf '%s\n' "$CMD" >> "$ROOT/command.txt"
eval "$CMD" 2>&1 | tee "$ROOT/logs/stage15_finalize_metrics.log"
