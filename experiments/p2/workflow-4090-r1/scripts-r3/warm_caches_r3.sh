#!/bin/bash
# Warm the three fresh r3 Jittor caches with the frozen controller references only (no
# candidate code), using exactly each TaskSpec's target env flags:
#   RAFT none/full: conv_opt=1 use_mkl=0 use_cutt=0 (no no_fuse)
#   FNO none:       additionally no_fuse=1 DISABLE_MULTIPROCESSING=1
set -uo pipefail
W=/root/autodl-tmp/workflow-4090-r1
LOG=$W/warm-r3.log
export nvcc_path=/usr/local/cuda-12.8/bin/nvcc conv_opt=1 use_mkl=0 use_cutt=0
export PATH=/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export python_config_path=/root/autodl-tmp/envs/target/bin/python3.9-config
PY=/root/autodl-tmp/envs/target/bin/python
warm() {
  export JITTOR_HOME=$1
  echo "=== warm $1 $(date -Is)" >> "$LOG"
  $PY "$2" 2>&1 | grep -Ev '^\[i ' | tail -6 >> "$LOG"
  echo "=== warm exit=${PIPESTATUS[0]}" >> "$LOG"
  find "$1" -name "jit_utils_core*.so" -o -name "jittor_core*.so" 2>/dev/null | sed 's#.*/##' | sort -u >> "$LOG"
  echo "cpython312=$(find "$1" -name '*cpython-312*' 2>/dev/null | wc -l)" >> "$LOG"
}
for c in none full; do
  warm /root/autodl-tmp/RAFT-Benchmark-v1/jittor-cache-wf-r3-$c $W/probe_raft_reference_r4.py
done
export no_fuse=1 DISABLE_MULTIPROCESSING=1
warm /root/autodl-tmp/FNO-Benchmark-v1/jittor-cache-wf-r3-none $W/probe_fno_reference.py
df -h /root/autodl-tmp | tail -1 >> "$LOG"
echo ALLDONE >> "$LOG"
