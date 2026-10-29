"""Freeze a small paired image/text batch sequence for a CLIP training pilot.

This is controller-side preparation. The source pilot already fixed ten CIFAR-10
images per class and one class prompt per class. Each training step takes one
image from every class, so the ten captions in a batch are distinct.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / "migrations/CLIP/data/pilot"
OUTPUT = ROOT / "migrations/CLIP/data/train-pilot"
STEPS = 4
CLASSES = 10


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    images = np.load(PILOT / "images.npy")
    tokens = np.load(PILOT / "tokens.npy")
    labels = np.load(PILOT / "labels.npy")
    indices = np.load(PILOT / "indices.npy")
    if images.shape != (100, 3, 224, 224) or tokens.shape != (CLASSES, 77):
        raise ValueError("CLIP pilot input shape changed")
    if labels.shape != (100,) or indices.shape != (100,):
        raise ValueError("CLIP pilot labels or indices changed")
    selected = np.stack([np.flatnonzero(labels == cls)[:STEPS]
                         for cls in range(CLASSES)], axis=1)
    if selected.shape != (STEPS, CLASSES):
        raise ValueError("Fewer than four pilot images for one class")
    if not np.all(labels[selected] == np.arange(CLASSES)[None, :]):
        raise ValueError("Each training step must contain one image per class")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    arrays = {
        "images.npy": np.ascontiguousarray(images[selected], dtype=np.float32),
        "tokens.npy": np.broadcast_to(tokens, (STEPS, *tokens.shape)).copy(),
        "source_indices.npy": indices[selected],
    }
    for name, array in arrays.items():
        np.save(OUTPUT / name, array, allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "purpose": "CLIP training feasibility pilot, not a frozen Benchmark v1 trial",
        "steps": STEPS,
        "batch_size": CLASSES,
        "selection": "step j uses the j-th fixed pilot image of each CIFAR-10 class",
        "source_sha256": {name: digest(PILOT / name)
                          for name in ("images.npy", "tokens.npy", "labels.npy", "indices.npy")},
        "output_sha256": {name: digest(OUTPUT / name) for name in arrays},
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "steps": STEPS,
                      "batch_size": CLASSES}, ensure_ascii=False))


if __name__ == "__main__":
    main()
