"""对照：PyTorch 在同样写法下的行为（判断差异是否属于 Jittor 特有）。"""

import torch

x = torch.randn(1, 2, 7, 3)
print("torch avg_pool2d kernel=(H,1) shape", tuple(torch.nn.functional.avg_pool2d(x, (7, 1)).shape))
try:
    torch.cat([torch.randn(3, 4), torch.randn(3, 2, 4)], dim=1)
    print("torch cat rank2+rank3 ok")
except Exception as exc:  # noqa: BLE001
    print("torch cat rank2+rank3 raised", type(exc).__name__)
t = torch.randn(1, 1, 3, 3)
print("tuple(torch.Tensor) of (1,1,3,3) len", len(tuple(t)), "item shape", tuple(tuple(t)[0].shape))
