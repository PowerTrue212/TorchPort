import numpy as np
import jittor as jt

print("jittor", jt.__version__, "has_cuda", bool(jt.has_cuda))
jt.flags.use_cuda = 1
print("flags use_cuda", jt.flags.use_cuda)
print("numpy", np.__version__)

# Module basics
class M(jt.Module):
    def __init__(self):
        self.w = jt.zeros((4, 4))
        self.bias = jt.array(np.ones((4,), dtype=np.float32))
    def execute(self, x):
        return jt.matmul(x, self.w) + self.bias

m = M()
print("has eval:", hasattr(m, "eval"), "has train:", hasattr(m, "train"))
try:
    m.eval()
    print("eval() ok")
except Exception as e:
    print("eval() failed:", e)
print("is jt.Module:", isinstance(m, jt.Module))

x = jt.array(np.random.randn(3, 4).astype(np.float32))
y = m(x)
print("matmul+bias ok:", y.shape)

# dtype checks
a = jt.array(np.random.randn(2, 3).astype(np.float32))
print("array dtype preserved:", a.dtype)

# slicing
print("x[0] shape:", x[0].shape)
print("x[:, 1:3] shape:", x[:, 1:3].shape)
xx = jt.array(np.random.randn(2, 3, 5).astype(np.float32))
print("xx[:, 0, :] shape:", xx[:, 0, :].shape)
print("xx[:, :, :2] shape:", xx[:, :, :2].shape)

# transpose / permute / reshape
print("transpose ok:", xx.transpose(0, 2, 1).shape)
try:
    print("permute ok:", xx.permute(2, 0, 1).shape)
except Exception as e:
    print("permute failed:", e)
print("reshape ok:", xx.reshape(2, -1).shape)

# softmax / sigmoid / sqrt / exp / argmax
s = jt.nn.softmax(xx, dim=-1)
print("softmax ok:", s.shape)
try:
    print("jt.nn.sigmoid:", jt.nn.sigmoid(xx).shape)
except Exception as e:
    print("jt.nn.sigmoid failed:", e)
try:
    print("jt.sigmoid ok:", jt.sigmoid(xx).shape)
except Exception as e:
    print("jt.sigmoid failed:", e)
try:
    print("x.sigmoid() ok:", xx.sigmoid().shape)
except Exception as e:
    print("x.sigmoid() failed:", e)
print("sqrt ok:", jt.sqrt(a).shape)
print("exp ok:", jt.exp(a).shape)
try:
    r = xx.argmax(dim=-1)
    print("argmax ok:", type(r), r)
except Exception as e:
    print("argmax failed:", e)

# broadcast add
b = jt.array(np.random.randn(5).astype(np.float32))
print("broadcast add ok:", (xx + b).shape)
c = jt.array(np.random.randn(2, 1, 5).astype(np.float32))
print("broadcast add2 ok:", (xx + c).shape)

# concat
print("concat ok:", jt.concat([xx, xx], dim=1).shape)

# expand / broadcast method
try:
    print("broadcast method:", b.broadcast((2, 3, 5)).shape)
except Exception as e:
    print("broadcast method failed:", e)
try:
    print("expand method:", b.expand((2, 3, 5)).shape)
except Exception as e:
    print("expand method failed:", e)

# matmul
print("matmul ok:", jt.matmul(xx, xx.transpose(0, 2, 1)).shape)

# gather
try:
    idx = jt.array(np.array([[0, 1], [1, 0]], dtype=np.int32))
    g = jt.gather(xx, 1, idx)
    print("gather ok:", g.shape)
except Exception as e:
    print("gather failed:", e)

# fancy indexing
try:
    fi = xx[jt.array(np.array([0, 1], dtype=np.int32))]
    print("fancy index ok:", fi.shape)
except Exception as e:
    print("fancy index failed:", e)

# nn modules
print("Conv2d:", jt.nn.Conv2d(3, 8, 3).execute if False else "exists")
conv = jt.nn.Conv2d(3, 8, 32, stride=32, bias=False)
print("conv weight shape:", conv.weight.shape)
im = jt.array(np.random.randn(2, 3, 224, 224).astype(np.float32))
print("conv out:", conv(im).shape)

lin = jt.nn.Linear(5, 7)
print("linear out:", lin(xx).shape)

emb = jt.nn.Embedding(10, 6)
tok = jt.array(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
print("embed out:", emb(tok).shape, emb(tok).dtype)

# layer norm
try:
    ln = jt.nn.LayerNorm(5)
    print("LayerNorm out:", ln(xx).shape)
except Exception as e:
    print("LayerNorm failed:", e)

# var.numpy
print("numpy ok:", y.numpy().shape, y.numpy().dtype)

# assign
try:
    w2 = jt.zeros((4, 4))
    w2.assign(jt.array(np.ones((4, 4), dtype=np.float32)))
    print("assign ok:", float(w2.sum()))
except Exception as e:
    print("assign failed:", e)

# sync
jt.sync_all(True)
print("sync_all ok")
print("DONE")
