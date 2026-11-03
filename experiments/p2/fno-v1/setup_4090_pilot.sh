#!/usr/bin/env bash
# Historical first pass. Its dependency resolution selected NumPy 2.2.6;
# use pin_4090_deps.sh for the source pilot dependency path.
set -euo pipefail
ROOT=/root/autodl-tmp/FNO-Benchmark-v1
ENVS=/root/autodl-tmp/envs
LOG=$ROOT/setup-logs
mkdir -p "$LOG" "$ROOT/deps"

"$ENVS/source/bin/python" -m pip install --no-cache-dir 'PyYAML==6.0.2' >"$LOG/source-pyyaml.log" 2>&1
"$ENVS/target/bin/python" -m pip install --no-cache-dir 'PyYAML==6.0.2' >"$LOG/target-pyyaml.log" 2>&1
"$ENVS/source/bin/python" -m pip install --no-cache-dir --target "$ROOT/deps" \
  'tensorly==0.9.0' 'tensorly-torch==0.5.0' 'opt-einsum==3.4.0' \
  'scipy==1.13.1' >"$LOG/fno-deps.log" 2>&1

test ! -e "$ROOT/source" || { echo 'source already exists; preserving it' >&2; exit 2; }
mkdir "$ROOT/source"
tar -xzf "$ROOT/materials/source-snapshot-a6648ee.tar.gz" -C "$ROOT/source"

export JITTOR_HOME=/root/autodl-tmp/jittor-cache-fno-smoke
export nvcc_path=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export conv_opt=1
"$ENVS/target/bin/python" - >"$LOG/target-smoke-convopt1.log" 2>&1 <<'PY'
import json
import numpy as np
import jittor as jt
jt.flags.use_cuda = 1
x = jt.array(np.random.default_rng(0).standard_normal((64,64)).astype('float32'))
y = (x @ x).numpy()
print(json.dumps({'jittor': jt.__version__, 'use_cuda': int(jt.flags.use_cuda), 'finite': bool(np.isfinite(y).all())}))
PY
"$ENVS/source/bin/python" -c 'import yaml; print(yaml.__version__)' >"$LOG/source-pyyaml-import.log"
"$ENVS/target/bin/python" -c 'import yaml; print(yaml.__version__)' >"$LOG/target-pyyaml-import.log"
