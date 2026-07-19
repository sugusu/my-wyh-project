#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wyh/TRGRGS;PY=/data/wyh/RecycleGS_Mainline/envs/tsgs_mainline/bin/python
"$PY" "$ROOT/tools/build_stage2s_fine_regions.py" | tee "$ROOT/logs/stage2s_build_fine_regions.log"
CUDA_VISIBLE_DEVICES=2 "$PY" "$ROOT/tools/run_stage2s_coarse_ablation.py" --split a --level fine 2>&1 | tee "$ROOT/logs/stage2s_fine_a.log" & pa=$!
CUDA_VISIBLE_DEVICES=3 "$PY" "$ROOT/tools/run_stage2s_coarse_ablation.py" --split b --level fine 2>&1 | tee "$ROOT/logs/stage2s_fine_b.log" & pb=$!
wait "$pa";wait "$pb"
