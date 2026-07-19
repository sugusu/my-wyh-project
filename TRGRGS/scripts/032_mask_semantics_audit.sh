#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wyh/TRGRGS;PY=/data/wyh/RecycleGS_Mainline/envs/tsgs_mainline/bin/python
echo "$PY $ROOT/tools/trace_transparency_mask.py" > "$ROOT/logs/stage1_p0_commands.txt"
$PY "$ROOT/tools/trace_transparency_mask.py" 2>&1 | tee "$ROOT/logs/stage1_p0_mask_semantics.log"

