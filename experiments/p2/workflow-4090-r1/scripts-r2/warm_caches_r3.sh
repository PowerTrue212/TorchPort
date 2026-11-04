#!/bin/bash
# Warm both per-condition Jittor caches with the frozen controller reference.
# usage: warm_caches_r3.sh <BENCH_DIR_NAME> <CACHE_PREFIX> <PROBE_SCRIPT>
set -uo pipefail
REPO=$1
PREFIX=$2
SCRIPT=$3
HOSTENV=/root/autodl-tmp/$REPO/$PREFIX-
export nvcc_path=/usr/local/cuda-12.8/bin/nvcc
export conv_opt=1 use_mkl=0 use_cutt=0 no_fuse=1 DISABLE_MULTIPROCESSING=1
export PATH=/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export python_config_path=/root/autodl-tmp/envs/target/bin/python3.9-config
for condition in none full; do
  export JITTOR_HOME=${HOSTENV}${condition}
  echo "=== warm $condition $JITTOR_HOME $(date -Is)"
  /root/autodl-tmp/envs/target/bin/python "$SCRIPT" 2>&1 | tail -3
  echo "=== warm $condition exit=$?"
  find "$JITTOR_HOME" -name "jit_utils_core*.so" -o -name "jittor_core*.so" 2>/dev/null | sed 's#.*/##' | sort -u
  find "$JITTOR_HOME" -name "cpython-312*" 2>/dev/null | head -2
done
echo ALLDONE
