#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/FNO-Benchmark-v1
export PYTHONPATH="$ROOT/deps-requests:$ROOT/deps-pinned:$ROOT/source"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for n in 1 2; do
  test ! -e "$ROOT/source-pilot-4090-r$n" || { echo "pilot r$n exists; preserving" >&2; exit 2; }
  /root/autodl-tmp/envs/source/bin/python "$ROOT/probe_source_pilot.py" \
    "$ROOT/source-pilot-4090-r$n" >"$ROOT/setup-logs/source-pilot-4090-r$n.log" 2>&1
done
