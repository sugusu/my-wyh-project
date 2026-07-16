#!/usr/bin/env bash
set -euo pipefail

OUT=/data/wyh/RecycleGS/outputs/debug/stage0rc/gpu_memory_trace.csv

mkdir -p "$(dirname "$OUT")"
echo "timestamp,index,memory_used_mib,memory_total_mib,gpu_util" > "$OUT"

while true
do
  nvidia-smi \
    --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F',' '$2 ~ /^ *[23] *$/ {
        gsub(/^ +| +$/, "", $1);
        gsub(/^ +| +$/, "", $2);
        gsub(/^ +| +$/, "", $3);
        gsub(/^ +| +$/, "", $4);
        gsub(/^ +| +$/, "", $5);
        print $1 "," $2 "," $3 "," $4 "," $5
      }' >> "$OUT"

  sleep 30
done
