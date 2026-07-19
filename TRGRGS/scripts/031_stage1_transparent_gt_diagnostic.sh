#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json
d=json.load(open('/data/wyh/TRGRGS/reports/stage1_gt_free_depth_sweep_r2.json'))
if d.get('status')!='PASS_GT_FREE_DEPTH_SWEEP':raise SystemExit('BLOCKED: GT-free Stage 1 did not pass; GT diagnostic must not run')
PY
ROOT=/data/wyh/TRGRGS
CMD="/data/wyh/RecycleGS_Mainline/envs/tsgs_mainline/bin/python $ROOT/tools/run_stage1_transparent_gt_diagnostic.py"
printf '%s\n' "$CMD" >> "$ROOT/command.txt"
eval "$CMD" 2>&1 | tee "$ROOT/logs/stage1_transparent_gt_diagnostic_r2.log"
