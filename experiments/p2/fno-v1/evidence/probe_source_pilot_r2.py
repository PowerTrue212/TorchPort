"""One-step FP32 FNO source pilot on the bundled Darcy 16x16 samples."""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from neuralop.models import FNO

root = Path(__file__).resolve().parent
data_path = root / "source/neuralop/data/datasets/data/darcy_train_16.pt"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)

torch.manual_seed(20261030)
torch.cuda.manual_seed_all(20261030)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

dataset = torch.load(data_path, map_location="cpu", weights_only=False)
x_cpu = dataset["x"][:2].unsqueeze(1).to(torch.float32).contiguous()
y_cpu = dataset["y"][:2].unsqueeze(1).to(torch.float32).contiguous()
np.savez_compressed(out / "inputs.npz", x=x_cpu.numpy(), y=y_cpu.numpy())

model = FNO(
    n_modes=(4, 4),
    in_channels=1,
    out_channels=1,
    hidden_channels=8,
    n_layers=2,
).cuda()
torch.save(model.state_dict(), out / "initial.pt")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x, y = x_cpu.cuda(), y_cpu.cuda()
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
prediction = model(x)
loss = torch.mean((prediction - y) ** 2)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

np.savez_compressed(
    out / "outputs.npz",
    prediction=prediction.detach().cpu().numpy(),
    loss=np.array(loss.detach().cpu().item(), dtype=np.float32),
)
torch.save(model.state_dict(), out / "post.pt")
result = {
    "status": "pilot_only",
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    "input_shape": list(x.shape),
    "output_shape": list(prediction.shape),
    "parameter_count": sum(p.numel() for p in model.parameters()),
    "loss": float(loss.detach().cpu().item()),
    "finite": bool(torch.isfinite(prediction).all() and torch.isfinite(loss)),
    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    "step_wall_seconds": elapsed,
}
(out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
