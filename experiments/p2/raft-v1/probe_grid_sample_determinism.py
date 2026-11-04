"""Check whether CUDA grid_sample backward is bitwise stable on this GPU."""
import os
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch  # noqa: E402

torch.manual_seed(7)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if "no-cudnn" in sys.argv:
    torch.backends.cudnn.enabled = False
x = torch.randn(256, 1, 48, 64, device="cuda")
grid = torch.rand(256, 9, 9, 2, device="cuda") * 2 - 1
grads = []
for _ in range(3):
    z = x.detach().clone().requires_grad_()
    y = torch.nn.functional.grid_sample(z, grid, align_corners=True)
    y.sum().backward()
    grads.append(z.grad.detach().clone())
print({"exact": [torch.equal(grads[0], item) for item in grads[1:]],
       "max_abs": [float((grads[0] - item).abs().max()) for item in grads[1:]]})
