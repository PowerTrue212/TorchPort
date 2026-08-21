#!/usr/bin/env bash
set -euo pipefail

QX_PYTHON="${QX_PYTHON:-python}"
DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-artifacts/jittor-deps}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

mkdir -p "${DEPENDENCY_ROOT}"

# Jittor 1.3.11.0 的数组桥接与 NumPy 2 不兼容，依赖安装在项目目录中，
# 不修改共享的 qx 虚拟环境。
PIP_CONFIG_FILE=/dev/null "${QX_PYTHON}" -m pip install \
  --index-url "${PIP_INDEX_URL}" \
  --target "${DEPENDENCY_ROOT}" \
  --no-deps \
  "numpy==1.26.4"
