"""Three-step RAFT source training preflight with official Sintel labels."""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/RAFT-Benchmark-v1")
sys.path.insert(0, str(ROOT / "deps"))
from PIL import Image  # noqa: E402

sys.path.insert(0, str(ROOT / "source/RAFT/core"))
from raft import RAFT  # noqa: E402


class Args(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def image(number: int) -> torch.Tensor:
    path = ROOT / f"data/sintel-subset/training/clean/alley_1/frame_{number:04d}.png"
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)[26:410, 256:768]
    assert arr.shape == (384, 512, 3)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1)[None].float().cuda()


def flow(number: int) -> torch.Tensor:
    path = ROOT / f"data/sintel-subset/training/flow/alley_1/frame_{number:04d}.flo"
    with path.open("rb") as stream:
        assert stream.read(4) == b"PIEH"
        width, height = struct.unpack("<ii", stream.read(8))
        arr = np.frombuffer(stream.read(), dtype="<f4").reshape(height, width, 2)
    assert (height, width) == (436, 1024)
    arr = arr[26:410, 256:768]
    return torch.from_numpy(arr.copy()).permute(2, 0, 1)[None].cuda()


def main() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(20261103)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    args = Args(small=False, mixed_precision=False, alternate_corr=False, dropout=0)
    model = RAFT(args).cuda().train()
    state = torch.load(ROOT / "checkpoints/raft-things.pth", map_location="cpu", weights_only=True)
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=5e-5, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 2e-5, total_steps=103, pct_start=0.05,
        cycle_momentum=False, anneal_strategy="linear")
    result = {"status": "pilot_only", "dataset": "MPI-Sintel training/clean/alley_1", "steps": []}
    for number in (2, 3, 4):
        a, b, gt = image(number), image(number + 1), flow(number)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(a, b, iters=12, test_mode=False)
        valid = (gt.square().sum(dim=1).sqrt() < 400).float()
        loss = sum(0.8 ** (11 - i) * (valid[:, None] * (pred - gt).abs()).mean()
                   for i, pred in enumerate(predictions))
        lr = optimizer.param_groups[0]["lr"]
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        result["steps"].append({"frame": number, "loss": float(loss), "lr": lr,
                                "grad_norm_before_clip": grad})
    out = ROOT / "evidence/source-train-r1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
