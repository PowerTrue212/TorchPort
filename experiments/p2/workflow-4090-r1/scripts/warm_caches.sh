#!/bin/bash
# Warm both per-condition Jittor caches with the frozen controller reference.
set -uo pipefail
REPO=$1
SUFFIX=$2
SCRIPT=$3
HOSTENV=/root/autodl-tmp/$REPO/jittor-cache-wf-r1-
export nvcc_path=/usr/local/cuda-12.8/bin/nvcc
export conv_opt=1 use_mkl=0 use_cutt=0
export PATH=/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export python_config_path=/root/autodl-tmp/envs/target/bin/python3.9-config
for condition in none full; do
  export JITTOR_HOME=${HOSTENV}${condition}
  echo "=== warm $condition $JITTOR_HOME $(date -Is)"
  /root/autodl-tmp/envs/target/bin/python "$SCRIPT" 2>&1 | tail -3
  echo "=== warm $condition exit=$?"
  ls "$JITTOR_HOME" 2>/dev/null | head -3
done
echo ALLDONE
