#!/usr/bin/env bash

# Copy this file to env.sh on the V100 host and adjust only when the paths differ.
export SOURCE_PYTHON="${SOURCE_PYTHON:-/data/L202500448/qx/miniconda3/envs/qx/bin/python}"
export JITTOR_PYTHON="${JITTOR_PYTHON:-/data/L202500448/torchport/tools/jittor-env/bin/python}"
export TORCHPORT_ROOT="${TORCHPORT_ROOT:-/data/L202500448/torchport}"
export CLIP_EXPERIMENT="${CLIP_EXPERIMENT:-${TORCHPORT_ROOT}/CLIP}"

# DSH + TorchPort only. Keep credentials in the process environment, never in this file.
export TORCHPORT_PYTHON="${TORCHPORT_PYTHON:-${SOURCE_PYTHON}}"
export TORCHPORT_PYTHONPATH="${TORCHPORT_PYTHONPATH:-${TORCHPORT_ROOT}/src}"
export TORCHPORT_TRACE_DIR="${TORCHPORT_TRACE_DIR:-${CLIP_EXPERIMENT}/results/dsh_torchport/traces}"
