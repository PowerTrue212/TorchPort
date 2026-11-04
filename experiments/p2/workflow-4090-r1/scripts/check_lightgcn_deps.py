"""Check that the LightGCN deps overlay has a numpy/pandas pair usable by torch 2.4.1."""
import sys

sys.path.insert(0, "/root/autodl-tmp/LightGCN-Benchmark-v1/deps")
import numpy  # noqa: E402
import pandas  # noqa: E402
import torch  # noqa: E402

print("numpy", numpy.__version__, "pandas", pandas.__version__, "torch", torch.__version__)
x = torch.arange(4.0)
print("numpy bridge", x.numpy(), numpy.asarray([1.0, 2.0]).mean())
