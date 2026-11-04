"""Independent official/custom backward training repeats and held-out EPE."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp/RAFT-Benchmark-v1")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path[:0] = [str(ROOT / "deps"), str(ROOT / "source/RAFT"), str(ROOT / "source/RAFT/core")]
from raft import RAFT  # noqa: E402
from source_adapter import Args, load_train_model  # noqa: E402

mode, run_id = sys.argv[1:3]
if mode not in ("official", "custom") or not run_id.isalnum():
    raise ValueError("mode or run_id")
out = ROOT / f"evidence/train-variant-{mode}-{run_id}"
out.mkdir(parents=True, exist_ok=False)
torch.manual_seed(20261103)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
args = Args(small=False, mixed_precision=False, alternate_corr=False, dropout=0)
model = (RAFT(args) if mode == "official" else load_train_model()).cuda().train()
checkpoint = torch.load(ROOT / "checkpoints/raft-things.pth", map_location="cpu", weights_only=True)
model.load_state_dict({key.removeprefix("module."): value for key, value in checkpoint.items()}, strict=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=5e-5, eps=1e-8)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 2e-5, total_steps=103, pct_start=0.05,
    cycle_momentum=False, anneal_strategy="linear")
inputs = [np.load(ROOT / f"inputs-r1/{name}.npy") for name in
          ("train-image1", "train-image2", "train-flow", "train-valid")]
result = {"status": "pilot_only", "mode": mode, "run_id": run_id, "steps": []}
for step in range(3):
    a, b, gt, valid = (torch.from_numpy(item[step]).cuda() for item in inputs)
    optimizer.zero_grad(set_to_none=True)
    pred = model(a, b, iters=12)
    magnitude = gt.square().sum(dim=1).sqrt()
    mask = (valid >= 0.5) & (magnitude < 400)
    loss = sum(0.8 ** (11 - i) * (mask[:, None] * (p - gt).abs()).mean()
               for i, p in enumerate(pred))
    lr = optimizer.param_groups[0]["lr"]
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize()
    np.savez_compressed(out / f"state-step-{step}.npz", **{
        key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
    result["steps"].append({"loss": float(loss), "lr": lr, "grad_norm_before_clip": grad_norm})

eval_samples = [(6, 7), (8, 9)]
if "--no-eval" not in sys.argv and all((ROOT / f"data/sintel-subset/training/clean/alley_1/frame_{n:04d}.png").is_file()
       for pair in eval_samples for n in pair):
    import struct

    from PIL import Image  # noqa: E402
    model.eval()
    with torch.no_grad():
        for first, second in eval_samples:
            def image(n):
                path = ROOT / f"data/sintel-subset/training/clean/alley_1/frame_{n:04d}.png"
                raw = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)[26:410, 256:768]
                return torch.from_numpy(raw.copy()).permute(2, 0, 1)[None].float().cuda()
            flow_path = ROOT / f"data/sintel-subset/training/flow/alley_1/frame_{first:04d}.flo"
            with flow_path.open("rb") as stream:
                assert stream.read(4) == b"PIEH"
                width, height = struct.unpack("<ii", stream.read(8))
                gt_array = np.frombuffer(stream.read(), dtype="<f4").reshape(height, width, 2)
            assert (height, width) == (436, 1024)
            gt_array = gt_array[26:410, 256:768]
            gt = torch.from_numpy(gt_array.copy()).permute(2, 0, 1)[None].cuda()
            _, prediction = model(image(first), image(second), iters=20, test_mode=True)
            epe = (prediction - gt).square().sum(dim=1).sqrt()
            mask = gt.square().sum(dim=1).sqrt() < 400
            result.setdefault("heldout_epe", {})[f"{first}-{second}"] = float(epe[mask].mean())
            np.save(out / f"heldout-flow-{first}-{second}.npy", prediction.cpu().numpy())
(out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
