"""Evaluate each saved post-training RAFT state on two held-out Sintel pairs."""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/RAFT-Benchmark-v1")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path[:0] = [str(ROOT / "deps"), str(ROOT / "source/RAFT"), str(ROOT / "source/RAFT/core")]
from PIL import Image  # noqa: E402
from raft import RAFT  # noqa: E402
from source_adapter import Args, load_train_model  # noqa: E402

mode, run_id = sys.argv[1:3]
path = ROOT / f"evidence/train-variant-{mode}-{run_id}"
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = (RAFT(Args(small=False, mixed_precision=False, alternate_corr=False, dropout=0))
         if mode == "official" else load_train_model()).cuda().eval()
with np.load(path / "state-step-2.npz") as archive:
    model.load_state_dict({key: torch.from_numpy(archive[key]) for key in archive.files}, strict=True)


def image(number):
    file = ROOT / f"data/sintel-subset/training/clean/alley_1/frame_{number:04d}.png"
    raw = np.asarray(Image.open(file).convert("RGB"), dtype=np.uint8)[26:410, 256:768]
    return torch.from_numpy(raw.copy()).permute(2, 0, 1)[None].float().cuda()


def ground_truth(number):
    file = ROOT / f"data/sintel-subset/training/flow/alley_1/frame_{number:04d}.flo"
    with file.open("rb") as stream:
        assert stream.read(4) == b"PIEH"
        width, height = struct.unpack("<ii", stream.read(8))
        raw = np.frombuffer(stream.read(), dtype="<f4").reshape(height, width, 2)
    assert (height, width) == (436, 1024)
    return torch.from_numpy(raw[26:410, 256:768].copy()).permute(2, 0, 1)[None].cuda()


result = {"status": "pilot_only", "mode": mode, "run_id": run_id, "heldout_epe": {}}
with torch.no_grad():
    for first, second in ((6, 7), (8, 9)):
        gt = ground_truth(first)
        _, pred = model(image(first), image(second), iters=20, test_mode=True)
        epe = (pred - gt).square().sum(dim=1).sqrt()
        valid = gt.square().sum(dim=1).sqrt() < 400
        key = f"{first}-{second}"
        result["heldout_epe"][key] = float(epe[valid].mean())
        np.save(path / f"heldout-flow-{key}.npy", pred.cpu().numpy())
(path / "heldout.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
