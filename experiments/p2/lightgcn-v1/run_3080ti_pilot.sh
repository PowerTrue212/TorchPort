#!/usr/bin/env bash
# Fresh hidden verifies under the separate repeat-envelope pilot contract.
set -euo pipefail
BASE=/root/autodl-tmp/LightGCN-Benchmark-v1
OUT=$BASE/controller/train-3080ti-r1
PY=/root/autodl-tmp/envs/source/bin/python
export PYTHONPATH=$BASE/harness/src

run_verify() {
  local task=$1 trial=$2
  echo "running-$trial"
  set +e
  "$PY" -m torchport verify "$OUT/$task" --root "$OUT/evidence" --trial "$trial" > "$OUT/verify-$trial.json" 2> "$OUT/verify-$trial.stderr.log"
  local code=$?
  set -e
  echo "$trial-exit-code=$code"
  "$PY" -m json.tool "$OUT/verify-$trial.json" >/dev/null
  if [[ $code != 0 && $code != 3 ]]; then exit "$code"; fi
}

run_verify task-pilot.json pilot-reference-holdout-1
run_verify task-pilot.json pilot-reference-holdout-2
run_verify task-pilot-neg-lr101.json pilot-neg-lr101
run_verify task-pilot-neg-reg101.json pilot-neg-reg101
echo 'pilot-verifies-complete'
