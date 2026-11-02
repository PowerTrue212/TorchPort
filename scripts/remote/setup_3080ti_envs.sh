#!/bin/bash
# Build isolated Benchmark v1 environments on the RTX 3080 Ti host. Everything lives under
# /root/autodl-tmp; the base miniconda install and system packages are not modified.
#   envs/source : Python 3.10, PyTorch 2.4.1 (PyPI wheel bundles CUDA 12.1 runtime)
#   envs/target : Python 3.9,  Jittor 1.3.11.0, nvcc from /usr/local/cuda-12.8
# The JTorch baseline env is added later, per repository, after its deps are frozen.
set -euo pipefail
ROOT=/root/autodl-tmp
LOG=$ROOT/setup-logs
mkdir -p "$ROOT/envs" "$LOG"
CONDA=/root/miniconda3/bin/conda

make_env() {
  local name=$1 py=$2
  if [ ! -x "$ROOT/envs/$name/bin/python" ]; then
    "$CONDA" create -y -p "$ROOT/envs/$name" "python=$py" >"$LOG/$name-create.log" 2>&1
  fi
}

make_env source 3.10
make_env target 3.9

"$ROOT/envs/source/bin/pip" install --no-cache-dir "torch==2.4.1" "numpy==1.26.4" \
  >"$LOG/source-pip.log" 2>&1
"$ROOT/envs/target/bin/pip" install --no-cache-dir "jittor==1.3.11.0" "numpy==1.26.4" \
  >"$LOG/target-pip.log" 2>&1

# Smoke: source CUDA matmul with TF32 off; target Jittor CUDA compile with an explicit cache.
"$ROOT/envs/source/bin/python" - >"$LOG/source-smoke.log" 2>&1 <<'PY'
import torch, json
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
x = torch.randn(64, 64, device="cuda")
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                  "device": torch.cuda.get_device_name(0),
                  "capability": torch.cuda.get_device_capability(0),
                  "finite": bool(torch.isfinite(x @ x).all())}))
PY
export JITTOR_HOME=$ROOT/jittor-cache-smoke nvcc_path=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
"$ROOT/envs/target/bin/python" - >"$LOG/target-smoke.log" 2>&1 <<'PY'
import json
import numpy as np
import jittor as jt
jt.flags.use_cuda = 1
x = jt.array(np.random.default_rng(0).standard_normal((64, 64)).astype("float32"))
y = (x @ x).numpy()
print(json.dumps({"jittor": jt.__version__, "use_cuda": int(jt.flags.use_cuda),
                  "finite": bool(np.isfinite(y).all())}))
PY
tail -1 "$LOG/source-smoke.log"
tail -1 "$LOG/target-smoke.log"
df -h "$ROOT" | tail -1
