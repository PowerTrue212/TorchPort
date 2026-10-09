import sys, numpy as np
x = np.random.default_rng(0).standard_normal((1, 4, 11, 13)).astype(np.float32)
if sys.argv[1] == "jt":
    import jittor as jt, inspect
    m = jt.nn.UpsamplingBilinear2d(scale_factor=2)
    print("jt attr align_corners", getattr(m, "align_corners", None))
    print(inspect.getsource(type(m).execute) if hasattr(type(m), "execute") else "")
    np.save("/tmp/up_jt.npy", m(jt.array(x)).numpy())
else:
    import torch
    t = torch.from_numpy(x)
    y = np.load("/tmp/up_jt.npy")
    for ac in (True, False):
        r = torch.nn.functional.interpolate(t, scale_factor=2, mode="bilinear", align_corners=ac).numpy()
        print("torch align_corners", ac, "max|jt-torch|", float(np.abs(y - r).max()))
    print("torch UpsamplingBilinear2d align_corners", torch.nn.UpsamplingBilinear2d(scale_factor=2).align_corners)
