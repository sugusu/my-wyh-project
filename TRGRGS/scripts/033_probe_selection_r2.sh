#!/usr/bin/env bash
set -euo pipefail
PY=/data/wyh/RecycleGS_Mainline/envs/tsgs_mainline/bin/python;ROOT=/data/wyh/TRGRGS
$PY "$ROOT/tools/freeze_probe_views_r2.py" 2>&1 | tee "$ROOT/logs/stage1_probe_selection_r2.log"

