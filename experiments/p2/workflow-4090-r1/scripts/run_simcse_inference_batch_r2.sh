#!/bin/bash
# SimCSE: second reference verify + negative control verify (serial, one GPU).
set -uo pipefail
R=/root/autodl-tmp/SimCSE-Benchmark-v1
cd "$R" || exit 1
mkdir -p "$R/inference-r1/evidence"
LOG=$R/inference-r1/evidence/batch-inference-r2.log
PY=/root/autodl-tmp/envs/source/bin/python
export PYTHONPATH=/root/autodl-tmp/workflow-4090-r1/src
echo "=== batch start $(date -Is)" >> "$LOG"
df -h /root/autodl-tmp | tail -1 >> "$LOG"
echo "=== smc-ref-d start $(date -Is)" >> "$LOG"
$PY inference-r1/scripts/run_inference_verify.py inference-r1/task-r4.json inference-r1/evidence smc-ref-d >> "$LOG" 2>&1
echo "=== smc-ref-d exit=$? $(date -Is)" >> "$LOG"
echo "=== smc-neg-2 start $(date -Is)" >> "$LOG"
$PY inference-r1/scripts/run_inference_verify.py inference-r1/task-r4.json inference-r1/evidence smc-neg-2 --repo inference-r1/neg-bias-r3 >> "$LOG" 2>&1
echo "=== smc-neg-2 exit=$? $(date -Is)" >> "$LOG"
echo "=== batch ALLDONE $(date -Is)" >> "$LOG"
