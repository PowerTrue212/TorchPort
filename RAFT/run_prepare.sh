#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${ROOT}/env.sh" ]]; then
  echo "Copy env.example.sh to env.sh and verify SOURCE_PYTHON first." >&2
  exit 1
fi
source "${ROOT}/env.sh"

"${SOURCE_PYTHON}" "${ROOT}/scripts/prepare_fixture.py"
"${SOURCE_PYTHON}" "${ROOT}/scripts/validate_protocol.py" --stage bootstrap
"${SOURCE_PYTHON}" "${ROOT}/scripts/prepare_reference.py"
"${SOURCE_PYTHON}" "${ROOT}/scripts/validate_protocol.py" --stage prepared
"${SOURCE_PYTHON}" "${ROOT}/scripts/init_workspaces.py"
"${SOURCE_PYTHON}" "${ROOT}/scripts/validate_protocol.py" --stage initialized
