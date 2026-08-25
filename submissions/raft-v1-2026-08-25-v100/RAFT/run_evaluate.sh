#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 codex|dsh|dsh_torchport" >&2
  exit 64
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT}/env.sh"
exec "${JITTOR_PYTHON}" "${ROOT}/scripts/evaluate_jittor.py" --method "$1"
