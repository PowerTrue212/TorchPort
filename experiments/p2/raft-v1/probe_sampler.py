import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch  # noqa: E402
from deterministic_sampler import (  # noqa: E402
    bilinear_sampler,
    official_forward_deterministic_backward,
)

torch.manual_seed(7)
torch.use_deterministic_algorithms(True)
image = torch.randn(32, 2, 12, 16, device="cuda")
coords = torch.rand(32, 9, 9, 2, device="cuda") * 20 - 2
outputs = []
grads = []
for _ in range(3):
    x = image.detach().clone().requires_grad_()
    y = bilinear_sampler(x, coords)
    y.sum().backward()
    outputs.append(y.detach().clone())
    grads.append(x.grad.detach().clone())
print("repeat", [torch.equal(grads[0], g) for g in grads[1:]])
reference_grid = torch.stack([2 * coords[..., 0] / 15 - 1,
                              2 * coords[..., 1] / 11 - 1], dim=-1)
ref = torch.nn.functional.grid_sample(image, reference_grid, align_corners=True)
print("forward max_abs", float((outputs[0] - ref).abs().max()))
official_outputs = []
official_grads = []
for _ in range(3):
    x = image.detach().clone().requires_grad_()
    y = official_forward_deterministic_backward(x, coords)
    y.sum().backward()
    official_outputs.append(y.detach().clone())
    official_grads.append(x.grad.detach().clone())
print("official forward exact", torch.equal(official_outputs[0], ref))
print("official forward repeat", [torch.equal(official_outputs[0], y) for y in official_outputs[1:]])
print("custom backward repeat", [torch.equal(official_grads[0], g) for g in official_grads[1:]])
