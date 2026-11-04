#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp/RAFT-Benchmark-v1
for name in inference-r1 train-three-r1; do
  bundle="$root/oracle-bundle-$name"
  test ! -e "$bundle" || { echo "preserving existing $bundle" >&2; exit 2; }
  mkdir "$bundle"
  tar -xzf "$root/oracle-bundle-$name.tar.gz" -C "$bundle"
  export PYTHONPATH="$bundle/src:$bundle/source:$root/deps"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  /root/autodl-tmp/envs/source/bin/python "$root/build_source_oracle.py" "$bundle" > "$root/evidence/oracle-$name.log" 2>&1
done
