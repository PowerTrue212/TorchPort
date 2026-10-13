import inspect
import numpy as np
import jittor as jt

print("Module.__init__:", inspect.signature(jt.Module.__init__))
print("Module.__setattr__:", inspect.signature(jt.Module.__setattr__))

jt.flags.use_cuda = 1
x = jt.array(np.random.randn(2, 3, 5).astype(np.float32))
try:
    m = x.mean(dim=-1, keepdims=True)
    print("mean keepdims ok:", m.shape)
except Exception as e:
    print("mean keepdims failed:", e)
try:
    s = x.sum(dim=1, keepdims=True)
    print("sum keepdims ok:", s.shape)
except Exception as e:
    print("sum keepdims failed:", e)
try:
    s2 = x.sum(dim=1)
    print("sum ok:", s2.shape)
except Exception as e:
    print("sum failed:", e)
try:
    sq = (x * x).sum(dim=-1, keepdims=True)
    print("x*x sum ok:", sq.shape)
except Exception as e:
    print("x*x failed:", e)
try:
    p = jt.pow(x, 2)
    print("pow ok:", p.shape)
except Exception as e:
    print("pow failed:", e)
try:
    z = jt.zeros(())
    e = jt.exp(z)
    print("0-dim exp ok:", e.shape, float(e.numpy()))
except Exception as e:
    print("0-dim failed:", e)
try:
    msk = np.full((5, 5), -np.inf, dtype=np.float32)
    msk = np.triu(msk, 1)
    mv = jt.array(msk)
    q = jt.array(np.random.randn(8, 5, 5).astype(np.float32))
    r = jt.nn.softmax(q + mv, dim=-1)
    print("mask add softmax ok:", r.shape, bool((r >= 0).all().numpy()))
except Exception as e:
    print("mask failed:", e)
try:
    a = jt.array(np.random.randn(2, 3).astype(np.float32))
    b = jt.array(np.random.randn(3, 4).astype(np.float32))
    c = jt.matmul(a, b)
    print("matmul2 ok:", c.shape)
    v = jt.array(np.array([1.0, 2.0], dtype=np.float32))
    print("scalar*mat ok:", (v[0] * c).shape)
except Exception as e:
    print("matmul2 failed:", e)
jt.sync_all(True)
print("DONE")
