#!/usr/bin/env bash

export SOURCE_PYTHON="${SOURCE_PYTHON:-/root/data-fs/RAFT/.venv-source/bin/python}"
export JITTOR_PYTHON="${JITTOR_PYTHON:-/root/miniconda3/envs/jittor/bin/python}"
export RAFT_EXPERIMENT="${RAFT_EXPERIMENT:-/root/data-fs/RAFT}"

export TORCHPORT_ROOT="${TORCHPORT_ROOT:-/root/data-fs/TorchPort}"
export TORCHPORT_PYTHON="${TORCHPORT_PYTHON:-${SOURCE_PYTHON}}"
export TORCHPORT_PYTHONPATH="${TORCHPORT_PYTHONPATH:-${TORCHPORT_ROOT}/src}"
export TORCHPORT_TRACE_DIR="${TORCHPORT_TRACE_DIR:-${RAFT_EXPERIMENT}/results/dsh_torchport/traces}"
