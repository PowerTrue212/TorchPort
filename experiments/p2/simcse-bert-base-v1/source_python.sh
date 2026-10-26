#!/bin/sh
set -eu
export PYTHONPATH="/root/SimCSE-Benchmark-v1/source-deps${PYTHONPATH:+:$PYTHONPATH}"
exec /root/data-fs/APT/.venv-source/bin/python "$@"
