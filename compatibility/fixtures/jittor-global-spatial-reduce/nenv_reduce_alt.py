"""全局空间归约的替代写法：能否在 Jittor 里得到确定且精确的结果。"""

import numpy as np

import jittor as jt

jt.flags.use_cuda = 1
d = np.load("nenv_in.npz")
ref = np.load("nenv_ref.npz")
x = jt.array(d["x_map"])
exact = ref["fp64/avgpool_global"].astype(np.float64).reshape(-1)
s = float(np.linalg.norm(exact))
b, c, h, w = x.shape


def variants():
    flat = x.reshape(b * c, h * w)
    ones = jt.ones((h * w, 1), dtype="float32") / (h * w)
    return {
        "adaptive_avgpool": jt.nn.AdaptiveAvgPool2d(1)(x),
        "mean_dims": x.mean(dims=(2, 3)),
        "matmul_ones": jt.matmul(flat, ones),
        "two_stage_mean": x.mean(dims=(3,)).mean(dims=(2,)),
    }


runs = []
with jt.no_grad():
    for _ in range(5):
        runs.append({k: v.numpy().astype(np.float64).reshape(-1) for k, v in variants().items()})
        jt.sync_all()
for name in runs[0]:
    reps = [r[name] for r in runs]
    n_env = max(np.linalg.norm(a - bb) / s for i, a in enumerate(reps) for bb in reps[i + 1:])
    err = max(np.linalg.norm(a - exact) / s for a in reps)
    print(f"{name:18s} N {n_env:.2e}  vs fp64 {err:.2e}  bitwise {all(np.array_equal(reps[0], r) for r in reps)}")
