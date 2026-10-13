"""Recompute the reference logits with the official PyTorch CLIP model.

Runs in the clip-source conda env. Reads the frozen prepared images/tokens and
the official checkpoint, evaluates the fp32 model on CUDA exactly like
scripts/prepare_reference.py, and saves the logits to the dsh workspace for the
smoke test (does NOT touch logits_ref.npy).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/data-fs/CLIP/source/openai-clip")
import clip  # noqa: E402

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = "cuda"
checkpoint = "/root/data-fs/CLIP/shared/checkpoints/ViT-B-32.pt"
model, _ = clip.load(checkpoint, device=device, jit=False)
model = model.float().eval()

images = np.load("/root/data-fs/CLIP/shared/prepared/images.npy", allow_pickle=False)
tokens = np.load("/root/data-fs/CLIP/shared/prepared/tokens.npy", allow_pickle=False)
images_t = torch.from_numpy(images).to(device)
tokens_t = torch.from_numpy(tokens).to(device)

with torch.inference_mode():
    logits_per_image, _ = model(images_t, tokens_t)

arr = logits_per_image.detach().cpu().numpy().astype(np.float32)
np.save("/root/data-fs/CLIP/workspaces/dsh/repo/my_ref_logits.npy", arr)
print("torch reference logits (fp32, V100):")
print(arr)
