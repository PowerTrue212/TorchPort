#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/tp-p2-final-20261003}"
ARCHIVE="${2:-/root/autodl-tmp/openai-clip-d05afc4.zip}"
COMMIT="d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
SOURCE="${ROOT}/CLIP/source"

rm -rf "${SOURCE}/openai-clip.clone-part" "${SOURCE}/openai-clip"
ROOT="${ROOT}" ARCHIVE="${ARCHIVE}" COMMIT="${COMMIT}" /root/miniconda3/bin/python - <<'PY'
import os
import shutil
import zipfile
from pathlib import Path

root = Path(os.environ["ROOT"])
archive = Path(os.environ["ARCHIVE"])
commit = os.environ["COMMIT"]
source = root / "CLIP" / "source"
with zipfile.ZipFile(archive) as package:
    package.extractall(source)
extracted = source / f"CLIP-{commit}"
destination = source / "openai-clip"
shutil.move(str(extracted), str(destination))
(destination / ".torchport_commit").write_text(commit + "\n", encoding="utf-8")
PY

mkdir -p "${ROOT}/CLIP/shared/checkpoints" "${ROOT}/CLIP/downloads"
CHECKPOINT="${ROOT}/CLIP/shared/checkpoints/ViT-B-32.pt"
CIFAR="${ROOT}/CLIP/downloads/cifar-10-python.tar.gz"
if ! echo "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af  ${CHECKPOINT}" | sha256sum -c --status; then
  curl -L --retry 3 --connect-timeout 30 -o "${CHECKPOINT}" \
    "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
fi
if ! echo "c58f30108f718f92721af3b95e74349a  ${CIFAR}" | md5sum -c --status; then
  curl -L --retry 3 --connect-timeout 30 -o "${CIFAR}" \
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
fi

cd "${ROOT}"
/root/miniconda3/bin/pip install 'ftfy<7' regex scipy
PYTHONPATH=CLIP/scripts /root/miniconda3/bin/python - <<'PY'
from pathlib import Path

from bootstrap import write_cifar_images
from common import load_config

config = load_config()
write_cifar_images(
    Path("CLIP/downloads/cifar-10-python.tar.gz"),
    config["input"]["samples"],
    Path("CLIP/shared/raw_images"),
)
PY

sha256sum CLIP/shared/checkpoints/ViT-B-32.pt
