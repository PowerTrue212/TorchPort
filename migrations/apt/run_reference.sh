#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT}/env.sh"
"${SOURCE_PYTHON}" "${ROOT}/scripts/validate_protocol.py" --stage initialized
"${SOURCE_PYTHON}" "${ROOT}/scripts/prepare_reference.py"
exec "${SOURCE_PYTHON}" "${ROOT}/scripts/validate_protocol.py" --stage reference

