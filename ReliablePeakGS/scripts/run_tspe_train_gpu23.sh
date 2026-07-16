#!/usr/bin/env bash
set -euo pipefail

source /data/wyh/ReliablePeakGS/scripts/stage0_env.sh
cd /data/wyh/ReliablePeakGS/external/TSPE-GS

exec python3 train.py "$@"
