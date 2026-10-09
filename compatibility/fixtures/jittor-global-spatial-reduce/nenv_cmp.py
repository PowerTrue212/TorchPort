"""汇总：每个算子的 Jittor 运行间波动 N_env、Jittor/PyTorch 相对 fp64 误差、PyTorch gpu/cpu 差（D 的单算子近似）。"""

import json

import numpy as np

ref = np.load("nenv_ref.npz")
jt = np.load("nenv_jt.npz")
names = sorted({k.split("/", 1)[1] for k in ref.files if k.startswith("fp64/")})
runs = sorted({k.split("/", 1)[0] for k in jt.files})


def rel(a, b, s):
    return float(np.sqrt(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2)) / s)


rows = {}
print(f"{'op':16s} {'N_env':>9s} {'jt-fp64':>9s} {'t32-fp64':>9s} {'D(gpu,cpu)':>10s} bitwise")
for n in names:
    exact = ref["fp64/" + n].astype(np.float64)
    s = float(np.sqrt(np.sum(exact ** 2))) or 1.0
    reps = [jt[f"{r}/{n}"].reshape(exact.shape) for r in runs]
    n_env = max(rel(a, b, s) for i, a in enumerate(reps) for b in reps[i + 1:])
    jt_err = max(rel(x, exact, s) for x in reps)
    t32 = rel(ref["t32/" + n], exact, s)
    d = rel(ref["t32/" + n], ref["cpu32/" + n], s)
    bitwise = all(np.array_equal(reps[0], x) for x in reps[1:])
    rows[n] = {"n_env": n_env, "jt_vs_fp64": jt_err, "torch32_vs_fp64": t32, "torch_gpu_vs_cpu": d, "bitwise": bitwise}
    print(f"{n:16s} {n_env:9.2e} {jt_err:9.2e} {t32:9.2e} {d:10.2e} {bitwise}")
json.dump(rows, open("nenv.json", "w"), indent=1)
