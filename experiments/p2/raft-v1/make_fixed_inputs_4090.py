"""Freeze RAFT source inputs and all checkpoint tensors for TorchPort tasks."""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/RAFT-Benchmark-v1")
sys.path.insert(0, str(ROOT / "deps"))
from PIL import Image  # noqa: E402

OUT = ROOT / "inputs-r1"
OUT.mkdir(exist_ok=False)


def image(path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    crop = raw[26:410, 256:768]
    assert crop.shape == (384, 512, 3)
    return np.ascontiguousarray(crop.transpose(2, 0, 1), dtype=np.float32)


def flow(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        assert stream.read(4) == b"PIEH"
        width, height = struct.unpack("<ii", stream.read(8))
        raw = np.frombuffer(stream.read(), dtype="<f4").reshape(height, width, 2)
    assert (height, width) == (436, 1024)
    return np.ascontiguousarray(raw[26:410, 256:768].transpose(2, 0, 1))


frames = [(ROOT / f"source/RAFT/demo-frames/frame_{n:04d}.png") for n in (16, 17)]
sintel = [ROOT / f"data/sintel-subset/training/clean/alley_1/frame_{n:04d}.png" for n in (2, 3)]
np.save(OUT / "inference-image1.npy", np.stack([image(frames[0]), image(sintel[0])]))
np.save(OUT / "inference-image2.npy", np.stack([image(frames[1]), image(sintel[1])]))
for name, offset in (("train-image1", 0), ("train-image2", 1)):
    np.save(OUT / f"{name}.npy", np.stack([
        image(ROOT / f"data/sintel-subset/training/clean/alley_1/frame_{n + offset:04d}.png")[None]
        for n in (2, 3, 4)]))
np.save(OUT / "train-flow.npy", np.stack([
    flow(ROOT / f"data/sintel-subset/training/flow/alley_1/frame_{n:04d}.flo")[None]
    for n in (2, 3, 4)]))
np.save(OUT / "train-valid.npy", np.ones((3, 1, 384, 512), dtype=np.float32))
checkpoint = torch.load(ROOT / "checkpoints/raft-things.pth", map_location="cpu", weights_only=True)
weights = {key.removeprefix("module."): value.numpy() for key, value in checkpoint.items()}
np.savez_compressed(OUT / "weights-fp32.npz", **weights)
manifest = {path.name: {"bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(OUT.iterdir()) if path.is_file()}
(ROOT / "evidence/fixed-inputs-r1.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
