import numpy as np

d = np.load("/root/data-fs/CLIP/shared/prepared/weights_fp32.npz")
keys = sorted(d.files)
print("num keys:", len(keys))
for k in keys:
    print(k, d[k].shape, d[k].dtype)

print("---- images/tokens ----")
images = np.load("/root/data-fs/CLIP/shared/prepared/images.npy", allow_pickle=False)
tokens = np.load("/root/data-fs/CLIP/shared/prepared/tokens.npy", allow_pickle=False)
print("images", images.shape, images.dtype, "min", images.min(), "max", images.max())
print("tokens", tokens.shape, tokens.dtype)
print(tokens)
print("argmax positions:", tokens.argmax(axis=1))
