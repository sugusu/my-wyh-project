#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wyh/TRGRGS
python3 - <<'PY'
import json
d=json.load(open('/data/wyh/TRGRGS/reports/stage15_final_decision.json'))
assert d['stage2a_authorized']
PY
CMD="/data/wyh/RecycleGS_Mainline/envs/tsgs_mainline/bin/python $ROOT/tools/run_stage2a_consensus.py"
printf '%s\n' "$CMD" >> "$ROOT/command.txt"
eval "$CMD" 2>&1 | tee "$ROOT/logs/stage2a_cross_view_consensus.log"
