#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SOURCE_PYTHON="${BASE_SOURCE_PYTHON:-/root/miniconda3/envs/clip-source/bin/python}"
VENV="${ROOT}/.venv-source"

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${BASE_SOURCE_PYTHON}" -m venv --system-site-packages "${VENV}"
fi
"${VENV}/bin/python" -m pip install --only-binary=:all: --no-deps \
  -r "${ROOT}/requirements/source.txt"
"${VENV}/bin/python" -m pip install --only-binary=:all: --no-deps \
  -r "${ROOT}/requirements/viz.txt"
"${VENV}/bin/python" -c \
  'import torch, numpy, PIL, scipy; print(torch.__version__, numpy.__version__, PIL.__version__, scipy.__version__)'
