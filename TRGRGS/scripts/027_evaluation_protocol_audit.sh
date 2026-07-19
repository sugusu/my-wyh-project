#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wyh/TRGRGS;PY=/data/wyh/RecycleGS_Mainline/envs/tsgs_mainline/bin/python;export CUDA_VISIBLE_DEVICES=2
{
 echo "$PY $ROOT/tools/trace_official_evaluator.py"
 echo "$PY $ROOT/tools/run_protocol_comparison_r3.py"
} > "$ROOT/logs/stage06_r3_commands.txt"
$PY "$ROOT/tools/trace_official_evaluator.py" 2>&1 | tee "$ROOT/logs/stage06_r3_trace.log"
$PY "$ROOT/tools/run_protocol_comparison_r3.py" 2>&1 | tee "$ROOT/logs/stage06_r3_protocols.log"
test "$(python3 -c 'import json; print(json.load(open("/data/wyh/TRGRGS/reports/stage06_r3_evaluation_protocol_equivalence.json"))["status"])')" = "PASS_PROTOCOL_EQUIVALENT"
$PY -m pytest -q "$ROOT/tests/test_official_metric_reproduction.py" "$ROOT/tests/test_stage1_r3_gate_template.py" | tee "$ROOT/logs/stage06_r3_pytest_targeted.log"
