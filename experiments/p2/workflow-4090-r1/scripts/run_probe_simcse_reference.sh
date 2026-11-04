#!/bin/bash
# Run the SimCSE controller reference probe with the frozen Jittor task environment.
set -uo pipefail
R=/root/autodl-tmp/SimCSE-Benchmark-v1
export JITTOR_HOME=$R/jittor-cache-inference-r1
export nvcc_path=/usr/local/cuda-12.8/bin/nvcc
export conv_opt=1 use_mkl=0 use_cutt=0
export PATH=/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export python_config_path=/root/autodl-tmp/envs/target/bin/python3.9-config
export PYTHONPATH=/root/autodl-tmp/workflow-4090-r1/src
cd "$R" || exit 1
/root/autodl-tmp/envs/target/bin/python inference-r1/scripts/probe_simcse_reference.py 2>&1 | tail -15
