#!/usr/bin/env bash
# Repeated frozen five-step warm-start runs for a separate judge-only baseline.
set -euo pipefail
BASE=/root/autodl-tmp/LightGCN-Benchmark-v1
OUT=$BASE/controller/train-ckpt-five-3080ti-r1
PY=/root/autodl-tmp/envs/source/bin/python
export PYTHONPATH=$BASE/harness/src

for index in $(seq 1 10); do
  echo "source-repeat-$index"
  "$PY" -m torchport run "$OUT/task.json" source --root "$OUT/evidence" --trial "source-repeat-$index" > "$OUT/source-repeat-$index.json" 2> "$OUT/source-repeat-$index.stderr.log"
  "$PY" -m json.tool "$OUT/source-repeat-$index.json" >/dev/null
done
for index in $(seq 1 10); do
  echo "target-repeat-$index"
  "$PY" -m torchport run "$OUT/task.json" target --root "$OUT/evidence" --trial "target-repeat-$index" > "$OUT/target-repeat-$index.json" 2> "$OUT/target-repeat-$index.stderr.log"
  "$PY" -m json.tool "$OUT/target-repeat-$index.json" >/dev/null
done
echo 'trained-five-repeats-complete'
