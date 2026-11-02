#!/usr/bin/env bash
# Collect raw negative-control target runs after the new reference has run.
set -euo pipefail
BASE=/root/autodl-tmp/LightGCN-Benchmark-v1
OUT=$BASE/controller/train-3080ti-r1
PY=/root/autodl-tmp/envs/source/bin/python
export PYTHONPATH=$BASE/harness/src

echo 'waiting-for-reference'
while ! grep -q '^reference-verify-complete$' "$OUT/run-reference.log" 2>/dev/null; do sleep 30; done
"$PY" - "$OUT/verify-reference-r1.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
if report.get('failure_code') not in (None, 'PARITY_FAILED'):
    raise SystemExit('Reference did not produce a comparable target run')
PY

for variant in target-neg-lr101-3080ti target-neg-reg101-3080ti; do
  echo "running-$variant"
  set +e
  "$PY" -m torchport verify "$OUT/task-$variant.json" --root "$OUT/evidence" --trial "$variant" > "$OUT/verify-$variant.json" 2> "$OUT/verify-$variant.stderr.log"
  code=$?
  set -e
  echo "$variant-exit-code=$code"
  "$PY" -m json.tool "$OUT/verify-$variant.json" >/dev/null
  if [[ $code != 0 && $code != 3 ]]; then exit "$code"; fi
done
echo 'negative-controls-complete'
