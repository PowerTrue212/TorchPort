#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/RAFT-Benchmark-v1
mode="$1"
count="$2"
start="${3:-1}"
for ((i=start; i<=count; i++)); do
  id="r$i"
  /root/autodl-tmp/envs/source/bin/python "$root/run_train_variant.py" "$mode" "$id" --no-eval > "$root/evidence/train-variant-$mode-$id.log" 2>&1
done
