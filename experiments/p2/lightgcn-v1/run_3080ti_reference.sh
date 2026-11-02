#!/usr/bin/env bash
# Run only the new 3080 Ti controller reference. Keep every raw result and log.
set -euo pipefail
BASE=/root/autodl-tmp/LightGCN-Benchmark-v1
SETUP=/root/autodl-tmp/setup-logs
OUT=$BASE/controller/train-3080ti-r1
PY=/root/autodl-tmp/envs/source/bin/python
export PYTHONPATH=$BASE/harness/src

echo 'waiting-for-smoke'
while pgrep -f '^bash setup_3080ti_envs.sh$' >/dev/null; do sleep 30; done
test -s "$SETUP/source-smoke.log"
test -s "$OUT/target-smoke-convopt1.log"
grep -q '"finite": true' "$SETUP/source-smoke.log"
grep -q '"finite": true' "$OUT/target-smoke-convopt1.log"
grep -q '"use_cuda": 1' "$OUT/target-smoke-convopt1.log"
echo 'smoke-passed'

# TorchPort's core CLI needs PyYAML in the source interpreter.
"$PY" -m pip install --no-cache-dir PyYAML==6.0.2 > "$OUT/source-pyyaml-install.log" 2>&1
"$PY" -c 'import yaml; print(yaml.__version__)'
echo 'torchport-dependency-passed'

while pgrep -f '[p]ip install.*LightGCN-Benchmark-v1/deps' >/dev/null; do sleep 30; done
PYTHONPATH="$BASE/deps:$BASE/harness/src" "$PY" -c 'import torch, pandas, scipy; print(torch.__version__, pandas.__version__, scipy.__version__)'
echo 'dependencies-passed'

"$PY" -m torchport inspect "$OUT/task.json" > "$OUT/inspect.json"
echo 'running-reference-verify'
set +e
"$PY" -m torchport verify "$OUT/task.json" --root "$OUT/evidence" --trial reference-r1 > "$OUT/verify-reference-r1.json" 2> "$OUT/verify-reference-r1.stderr.log"
code=$?
set -e
echo "reference-verify-exit-code=$code"
"$PY" -m json.tool "$OUT/verify-reference-r1.json" >/dev/null
if [[ $code != 0 && $code != 3 ]]; then exit "$code"; fi
echo 'reference-verify-complete'
