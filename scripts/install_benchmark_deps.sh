#!/usr/bin/env bash
set -euo pipefail

QX_PYTHON="${QX_PYTHON:-python}"
DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-artifacts/python-deps}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

mkdir -p "${DEPENDENCY_ROOT}"

PIP_CONFIG_FILE=/dev/null "${QX_PYTHON}" -m pip install \
  --index-url "${PIP_INDEX_URL}" \
  --target "${DEPENDENCY_ROOT}" \
  --no-deps \
  "timm==1.0.28"
