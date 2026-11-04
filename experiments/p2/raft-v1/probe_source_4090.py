"""Pilot source inference repeatability on the fixed RAFT checkpoint and frame pairs."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/RAFT-Benchmark-v1/deps")
from PIL import Image

ROOT = Path("/root/autodl-tmp/RAFT-Benchmark-v1")
sys.path.insert(0, str(ROOT / "source/RAFT/core"))
from raft import RAFT  # noqa: E402


class Args(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def tensor(path: Path) -> torch.Tensor:
    raw = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    raw = raw[26:410, 256:768]
    assert raw.shape == (384, 512, 3), (path, raw.shape)
    return torch.from_numpy(raw.copy()).permute(2, 0, 1)[None].float().cuda()


def main() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args = Args(small=False, mixed_precision=False, alternate_corr=False, dropout=0)
    model = RAFT(args)
    ckpt = torch.load(ROOT / "checkpoints/raft-things.pth", map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in ckpt.items()}
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    pairs = {
        "demo": [ROOT / "source/RAFT/demo-frames/frame_0016.png",
                 ROOT / "source/RAFT/demo-frames/frame_0017.png"],
        "sintel": [ROOT / "data/sintel-subset/training/clean/alley_1/frame_0002.png",
                   ROOT / "data/sintel-subset/training/clean/alley_1/frame_0003.png"],
    }
    out = ROOT / "evidence/source-inference-r1"
    out.mkdir(parents=True, exist_ok=True)
    result = {"status": "pilot_only", "iters": 20, "pairs": {}}
    for name, paths in pairs.items():
        if not all(path.is_file() for path in paths):
            continue
        a, b = (tensor(path) for path in paths)
        runs = []
        with torch.no_grad():
            for _ in range(2):
                low, up = model(a, b, iters=20, test_mode=True)
                torch.cuda.synchronize()
                runs.append((low.cpu().numpy(), up.cpu().numpy()))
        exact = all(np.array_equal(runs[0][i], runs[1][i]) for i in range(2))
        np.savez_compressed(out / f"{name}-outputs.npz", flow_low=runs[0][0], flow_up=runs[0][1])
        np.savez_compressed(out / f"{name}-inputs.npz", image1=a.cpu().numpy(), image2=b.cpu().numpy())
        result["pairs"][name] = {
            "exact_repeat": exact,
            "shapes": [list(value.shape) for value in runs[0]],
            "output_sha256": hashlib.sha256((out / f"{name}-outputs.npz").read_bytes()).hexdigest(),
        }
        if not exact:
            raise RuntimeError(f"source inference not bitwise repeatable: {name}")
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
