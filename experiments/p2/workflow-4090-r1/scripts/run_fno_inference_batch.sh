#!/bin/bash
# FNO inference: second reference verify + one negative control verify (serial, one GPU).
set -uo pipefail
R=/root/autodl-tmp/FNO-Benchmark-v1
cd "$R" || exit 1
LOG=$R/inference-r1/evidence/batch-inference-r1.log
PY=/root/autodl-tmp/envs/source/bin/python
export PYTHONPATH=$R/oracle-bundle-r3/src
echo "=== batch start $(date -Is)" >> "$LOG"
df -h /root/autodl-tmp | tail -1 >> "$LOG"
echo "=== fno-ref-3 start $(date -Is)" >> "$LOG"
$PY inference-r1/scripts/run_inference_verify.py inference-r1/task-r2.json inference-r1/evidence fno-ref-3 >> "$LOG" 2>&1
echo "=== fno-ref-3 exit=$? $(date -Is)" >> "$LOG"
echo "=== fno-neg-1 start $(date -Is)" >> "$LOG"
$PY inference-r1/scripts/run_inference_verify.py inference-r1/task-r2.json inference-r1/evidence fno-neg-1 --repo inference-r1/neg-bias-001 >> "$LOG" 2>&1
echo "=== fno-neg-1 exit=$? $(date -Is)" >> "$LOG"
echo "=== batch ALLDONE $(date -Is)" >> "$LOG"
