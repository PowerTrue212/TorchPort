#!/bin/bash
# SimCSE inference: two reference verifies + one negative control verify (serial, one GPU).
set -uo pipefail
R=/root/autodl-tmp/SimCSE-Benchmark-v1
cd "$R" || exit 1
LOG=$R/inference-r1/evidence/batch-inference-r1.log
PY=/root/autodl-tmp/envs/source/bin/python
export PYTHONPATH=/root/autodl-tmp/workflow-4090-r1/src
echo "=== batch start $(date -Is)" >> "$LOG"
df -h /root/autodl-tmp | tail -1 >> "$LOG"
for trial in smc-ref-1 smc-ref-2; do
  echo "=== $trial start $(date -Is)" >> "$LOG"
  $PY inference-r1/scripts/run_inference_verify.py inference-r1/task-r1.json inference-r1/evidence "$trial" >> "$LOG" 2>&1
  echo "=== $trial exit=$? $(date -Is)" >> "$LOG"
done
echo "=== smc-neg-1 start $(date -Is)" >> "$LOG"
$PY inference-r1/scripts/run_inference_verify.py inference-r1/task-r1.json inference-r1/evidence smc-neg-1 --repo inference-r1/neg-bias-001 >> "$LOG" 2>&1
echo "=== smc-neg-1 exit=$? $(date -Is)" >> "$LOG"
echo "=== batch ALLDONE $(date -Is)" >> "$LOG"
