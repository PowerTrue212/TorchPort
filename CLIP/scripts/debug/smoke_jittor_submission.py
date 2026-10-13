"""Smoke test mirroring scripts/evaluate_jittor.py for the dsh submission.

Runs in the jittor conda env. Compares the migrated logits against the torch
reference recomputed into my_ref_logits.npy (workspace-local, not
logits_ref.npy) with the frozen tolerances.
"""
import importlib.util
import numpy as np
import jittor as jt

print("jittor", jt.__version__, "numpy", np.__version__, "cuda", bool(jt.has_cuda))
jt.flags.use_cuda = 1

spec = importlib.util.spec_from_file_location(
    "clip_migration_submission",
    "/root/data-fs/CLIP/workspaces/dsh/repo/submission.py",
)
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

images = np.load("/root/data-fs/CLIP/shared/prepared/images.npy", allow_pickle=False)
tokens = np.load("/root/data-fs/CLIP/shared/prepared/tokens.npy", allow_pickle=False)

model = sub.load_model("/root/data-fs/CLIP/shared/prepared/weights_fp32.npz")
print("model type:", type(model), "is jt.Module:", isinstance(model, jt.Module))
model.eval()
with jt.no_grad():
    logits = sub.forward_logits(model, images, tokens)
    print("logits type:", type(logits), "is jt.Var:", isinstance(logits, jt.Var))
    jt.sync_all(True)
    arr = np.asarray(logits.numpy(), dtype=np.float32)

print("jittor logits:")
print(arr)
print("shape:", arr.shape)

ref = np.load("/root/data-fs/CLIP/workspaces/dsh/repo/my_ref_logits.npy")
diff = np.abs(ref.astype(np.float64) - arr.astype(np.float64))
print("torch reference logits:")
print(ref)
print("max abs error:", float(diff.max()))
print("allclose rtol=1e-4 atol=1e-4:", bool(np.allclose(ref, arr, rtol=1e-4, atol=1e-4)))
