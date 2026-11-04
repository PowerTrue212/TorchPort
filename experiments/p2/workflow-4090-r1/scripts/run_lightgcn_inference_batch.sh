#!/bin/bash
# LightGCN inference: second reference verify + one negative control verify (serial).
set -uo pipefail
R=/root/autodl-tmp/LightGCN-Benchmark-v1
cd "$R" || exit 1
LOG=$R/inference-r1/evidence/batch-inference-r1.log
PY=/root/autodl-tmp/envs/source/bin/python
export PYTHONPATH=/root/autodl-tmp/workflow-4090-r1/src
echo "=== batch start $(date -Is)" >> "$LOG"
df -h /root/autodl-tmp | tail -1 >> "$LOG"
echo "=== lgcn-ref-b start $(date -Is)" >> "$LOG"
$PY inference-r1/scripts/run_inference_verify.py inference-r1/task-r1.json inference-r1/evidence lgcn-ref-b >> "$LOG" 2>&1
echo "=== lgcn-ref-b exit=$? $(date -Is)" >> "$LOG"
echo "=== lgcn-neg-1 start $(date -Is)" >> "$LOG"
$PY inference-r1/scripts/run_inference_verify.py inference-r1/task-r1.json inference-r1/evidence lgcn-neg-1 --repo inference-r1/neg-bias-001 >> "$LOG" 2>&1
echo "=== lgcn-neg-1 exit=$? $(date -Is)" >> "$LOG"
echo "=== batch ALLDONE $(date -Is)" >> "$LOG"
