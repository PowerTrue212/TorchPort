"""N_env 探测（Jittor 端）：同输入同权重，每个算子独立进程内重复 R 次，存 nenv_jt.npz。"""

import numpy as np

import jittor as jt

jt.flags.use_cuda = 1
R = 5
d = np.load("nenv_in.npz")
g = {k: jt.array(d[k]) for k in d.files}
nn = jt.nn


def ops():
    att = jt.matmul(g["q"], g["k"].transpose(0, 1, 3, 2)) * 0.125
    sm = nn.softmax(att, dim=-1)
    return {
        "conv_k3s1": nn.conv2d(g["x_map"], g["w_k3"], g["b_k3"], padding=1),
        "conv_k3s1_d6": nn.conv2d(g["x_map"], g["w_k3"], g["b_k3"], padding=6, dilation=6),
        "conv_k1": nn.conv2d(g["x_map"], g["w_k1"], g["b_k1"]),
        "conv_k8s8": nn.conv2d(g["x_map"], g["w_k8"], g["b_k8"], stride=8),
        "avgpool_global": nn.AdaptiveAvgPool2d(1)(g["x_map"]),
        "mean_lastdim": g["x_tok"].mean(-1),
        "sum_spatial": g["x_map"].sum(dims=(2, 3)),
        "linear": nn.matmul_transpose(g["x_tok"], g["w_lin"]) + g["b_lin"],
        "matmul_qk": att,
        "softmax": sm,
        "attn_av": jt.matmul(sm, g["v"]),
        "layernorm": nn.layer_norm(g["x_tok"], (64,), g["ln_w"], g["ln_b"], 1e-6),
        "gelu": nn.gelu(g["x_tok"]),
        "interp_up2": nn.interpolate(g["x_small"], scale_factor=2, mode="bilinear", align_corners=False),
    }


out = {}
with jt.no_grad():
    for i in range(R):
        for name, val in ops().items():
            out[f"r{i}/{name}"] = val.numpy()
        jt.sync_all()
        jt.gc()
np.savez("nenv_jt.npz", **out)
print("saved", len(out))
