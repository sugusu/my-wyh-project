#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wyh/TRGRGS
PYTHON=/data/wyh/RecycleGS_Mainline/envs/tsgs_mainline/bin/python
GPU="${TRGRGS_GPU:-3}"
case "$GPU" in 2|3) ;; *) echo "ERROR: only physical GPUs 2 and 3 are allowed" >&2; exit 64;; esac
CMD=("$PYTHON" "$ROOT/tools/coordinate_audit.py" --config "$ROOT/configs/scene01_dev.yaml" --scale 0.25)
printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU" > "$ROOT/outputs/scene_01/coordinate_audit/command.txt"
printf '%q ' "${CMD[@]}" >> "$ROOT/outputs/scene_01/coordinate_audit/command.txt"; printf '\n' >> "$ROOT/outputs/scene_01/coordinate_audit/command.txt"
CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" 2>&1 | tee "$ROOT/logs/scene01_coordinate_audit.log"
