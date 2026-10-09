"""N_env 探测（PyTorch 端）：生成固定输入/权重，算 fp64 真值与 fp32 结果，存 nenv_ref.npz。

形状取自 PlantCamo（704 输入，stage1 176x176x64，PVT block1 N=30976, sr=8 -> 484）。
"""

import numpy as np
import torch

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
rng = np.random.default_rng(0)


def r(*shape, scale=1.0):
    return (scale * rng.standard_normal(shape)).astype(np.float32)


data = {
    "x_map": r(1, 64, 176, 176),
    "w_k3": r(64, 64, 3, 3, scale=0.05), "b_k3": r(64, scale=0.1),
    "w_k1": r(16, 64, 1, 1, scale=0.1), "b_k1": r(16, scale=0.1),
    "w_k8": r(64, 64, 8, 8, scale=0.02), "b_k8": r(64, scale=0.1),
    "x_tok": r(1, 30976, 64),
    "w_lin": r(256, 64, scale=0.1), "b_lin": r(256, scale=0.1),
    "q": r(1, 1, 30976, 64), "k": r(1, 1, 484, 64), "v": r(1, 1, 484, 64),
    "ln_w": r(64, scale=0.5) + 1, "ln_b": r(64, scale=0.1),
    "x_small": r(1, 64, 44, 44),
}


def ops(t, dev, dtype):
    F = torch.nn.functional
    g = {k: torch.from_numpy(v).to(dev, dtype) for k, v in data.items()}
    with torch.no_grad():
        att = (g["q"] @ g["k"].transpose(-2, -1)) * 0.125
        return {
            "conv_k3s1": F.conv2d(g["x_map"], g["w_k3"], g["b_k3"], padding=1),
            "conv_k3s1_d6": F.conv2d(g["x_map"], g["w_k3"], g["b_k3"], padding=6, dilation=6),
            "conv_k1": F.conv2d(g["x_map"], g["w_k1"], g["b_k1"]),
            "conv_k8s8": F.conv2d(g["x_map"], g["w_k8"], g["b_k8"], stride=8),
            "avgpool_global": F.adaptive_avg_pool2d(g["x_map"], 1),
            "mean_lastdim": g["x_tok"].mean(-1),
            "sum_spatial": g["x_map"].sum(dim=(2, 3)),
            "linear": F.linear(g["x_tok"], g["w_lin"], g["b_lin"]),
            "matmul_qk": att,
            "softmax": att.softmax(-1),
            "attn_av": att.softmax(-1) @ g["v"],
            "layernorm": F.layer_norm(g["x_tok"], (64,), g["ln_w"], g["ln_b"], 1e-6),
            "gelu": F.gelu(g["x_tok"]),
            "interp_up2": F.interpolate(g["x_small"], scale_factor=2, mode="bilinear", align_corners=False),
        }


out = {}
for name, val in ops(None, "cuda", torch.float64).items():
    out["fp64/" + name] = val.cpu().numpy()
for name, val in ops(None, "cuda", torch.float32).items():
    out["t32/" + name] = val.cpu().numpy()
for name, val in ops(None, "cpu", torch.float32).items():
    out["cpu32/" + name] = val.numpy()
np.savez("nenv_in.npz", **data)
np.savez("nenv_ref.npz", **out)
print("saved", len(out))
