#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/FNO-Benchmark-v1
mkdir -p "$ROOT/deps-pinned" "$ROOT/setup-logs"
/root/autodl-tmp/envs/source/bin/python -m pip install --no-cache-dir \
  --target "$ROOT/deps-pinned" \
  'numpy==1.26.4' 'scipy==1.13.1' 'tensorly==0.9.0' \
  'tensorly-torch==0.5.0' 'opt-einsum==3.4.0' \
  >"$ROOT/setup-logs/fno-deps-pinned.log" 2>&1
mkdir -p "$ROOT/deps-requests"
/root/autodl-tmp/envs/source/bin/python -m pip install --no-cache-dir \
  --target "$ROOT/deps-requests" 'requests==2.34.2' \
  >"$ROOT/setup-logs/fno-requests.log" 2>&1
PYTHONPATH="$ROOT/deps-requests:$ROOT/deps-pinned:$ROOT/source" \
  /root/autodl-tmp/envs/source/bin/python -c \
  'import numpy, scipy, tensorly, tltorch, neuralop; print(numpy.__version__, scipy.__version__, tensorly.__version__)' \
  >"$ROOT/setup-logs/fno-import-pinned.log" 2>&1
