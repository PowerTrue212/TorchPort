#!/usr/bin/env bash

APT_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export APT_EXPERIMENT="${APT_EXPERIMENT:-${APT_ENV_DIR}}"
export TORCHPORT_ROOT="${TORCHPORT_ROOT:-$(cd "${APT_EXPERIMENT}/.." && pwd)}"

# Set these to the isolated PyTorch and Jittor interpreters on the V100 host.
export SOURCE_PYTHON="${SOURCE_PYTHON:-python}"
export JITTOR_PYTHON="${JITTOR_PYTHON:-python}"

# DSH + TorchPort only. Never put credentials in this file.
export TORCHPORT_PYTHON="${TORCHPORT_PYTHON:-${SOURCE_PYTHON}}"
export TORCHPORT_PYTHONPATH="${TORCHPORT_PYTHONPATH:-${TORCHPORT_ROOT}/src}"
export TORCHPORT_TRACE_DIR="${TORCHPORT_TRACE_DIR:-${APT_EXPERIMENT}/results/dsh_torchport/traces}"
unset APT_ENV_DIR

