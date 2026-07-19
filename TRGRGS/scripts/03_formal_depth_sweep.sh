#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wyh/TRGRGS
python3 - <<'PY'
import json
p=json.load(open('/data/wyh/TRGRGS/outputs/scene_01/probe_views_r2.json'))
a=json.load(open('/data/wyh/TRGRGS/reports/stage1_p0_mask_semantics_audit.json'))
if a.get('status')!='PASS_MASK_RENDERER_EQUIVALENCE':raise SystemExit('BLOCKED: mask equivalence did not pass')
if p.get('status') not in ('PASS_GT_FREE_PROBE_SELECTION','PASS_LIMITED_CANDIDATE_POOL') or len(p.get('probe_ids',[]))!=8:raise SystemExit('BLOCKED: eight GT-free R2 probes were not frozen')
PY
mkdir -p "$ROOT/logs"
CMD="CUDA_VISIBLE_DEVICES=2 /data/wyh/RecycleGS_Mainline/envs/tsgs_mainline/bin/python $ROOT/tools/run_formal_depth_sweep.py"
printf '%s\n' "$CMD" >> "$ROOT/command.txt"
eval "$CMD" 2>&1 | tee "$ROOT/logs/stage1_formal_depth_sweep_r2.log"
