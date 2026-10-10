"""VSCode-v2 Agent 报告的 Jittor 约束：逐条最小复现，与 numpy 真值比较。只打印事实，不下结论。"""

import numpy as np

import jittor as jt

jt.flags.use_cuda = 1
rng = np.random.default_rng(0)


def show(name, got, want):
    got = np.asarray(got)
    same_shape = got.shape == np.asarray(want).shape
    err = float(np.abs(got - want).max()) if same_shape else None
    print(f"{name:40s} shape {got.shape} want {np.asarray(want).shape} max_abs {err}")


# 1. matmul：2-D x 2-D、3-D x 2-D、3-D x 3-D
a2, b2 = rng.standard_normal((5, 4)).astype(np.float32), rng.standard_normal((4, 3)).astype(np.float32)
show("matmul 2Dx2D", jt.matmul(jt.array(a2), jt.array(b2)).numpy(), a2 @ b2)
a3 = rng.standard_normal((2, 5, 4)).astype(np.float32)
show("matmul 3Dx2D", jt.matmul(jt.array(a3), jt.array(b2)).numpy(), a3 @ b2)
b3 = rng.standard_normal((2, 4, 3)).astype(np.float32)
show("matmul 3Dx3D", jt.matmul(jt.array(a3), jt.array(b3)).numpy(), a3 @ b3)
a4 = rng.standard_normal((2, 6, 5, 4)).astype(np.float32)
b4 = rng.standard_normal((2, 6, 4, 3)).astype(np.float32)
show("matmul 4Dx4D", jt.matmul(jt.array(a4), jt.array(b4)).numpy(), a4 @ b4)

# 2. avg_pool2d 窗口等于输入（VSCode 源码写法 F.avg_pool2d(x, (HW, 1))）
x = rng.standard_normal((1, 2, 7, 3)).astype(np.float32)
try:
    got = jt.nn.avg_pool2d(jt.array(x), (7, 1)).numpy()
    show("avg_pool2d kernel=(H,1)", got, x.mean(axis=2, keepdims=True))
except Exception as exc:  # noqa: BLE001
    print(f"{'avg_pool2d kernel=(H,1)':40s} raised {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")

# 3. concat 不同秩（torch.cat 同样要求同秩，这里只记录 Jittor 行为）
p, q = rng.standard_normal((3, 4)).astype(np.float32), rng.standard_normal((3, 2, 4)).astype(np.float32)
try:
    got = jt.concat([jt.array(p), jt.array(q)], dim=1).numpy()
    print(f"{'concat rank2+rank3 dim=1':40s} shape {got.shape}")
except Exception as exc:  # noqa: BLE001
    print(f"{'concat rank2+rank3 dim=1':40s} raised {type(exc).__name__}")

# 4. repeat_interleave 后 [..., 0, :] 索引
r = rng.standard_normal((2, 3, 4)).astype(np.float32)
rep = np.repeat(r, 2, axis=1)
try:
    jr = jt.array(r)
    got = (jr.repeat_interleave(2, dim=1) if hasattr(jr, "repeat_interleave") else jt.misc.repeat_interleave(jr, 2, dim=1))
    show("repeat_interleave dim=1", got.numpy(), rep)
    show("repeat_interleave then [...,0,:]", got[..., 0, :].numpy(), rep[..., 0, :])
except Exception as exc:  # noqa: BLE001
    print(f"{'repeat_interleave':40s} raised {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")

# 5. tuple(Var) 沿第 0 维切分（与 tuple(torch.Tensor) 行为对照见 claims_torch.py）
v = jt.array(rng.standard_normal((1, 1, 3, 3)).astype(np.float32))
print(f"{'tuple(Var) of (1,1,3,3)':40s} len {len(tuple(v))} item shape {tuple(v)[0].shape}")
