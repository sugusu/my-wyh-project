#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wyh/TRGRGS;PY=/data/wyh/RecycleGS_Mainline/envs/tsgs_mainline/bin/python
test "$(python3 -c "import json;print(json.load(open('$ROOT/reports/stage2s_split_model_parity.json'))['status'])")" = PASS_SPLIT_MODEL_PARITY
CUDA_VISIBLE_DEVICES=2 "$PY" "$ROOT/tools/run_stage2s_coarse_ablation.py" --split a 2>&1 | tee "$ROOT/logs/stage2s_coarse_a.log" & pa=$!
CUDA_VISIBLE_DEVICES=3 "$PY" "$ROOT/tools/run_stage2s_coarse_ablation.py" --split b 2>&1 | tee "$ROOT/logs/stage2s_coarse_b.log" & pb=$!
wait "$pa";wait "$pb"
