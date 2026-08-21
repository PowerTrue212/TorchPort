from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[2]
data = np.load(root / "data" / "tensors" / "vit_mae_tensors.npz")

panels = [
    ("PyTorch", "ViT attention", data["vit_attn_src"], "viridis"),
    ("Jittor", "ViT attention", data["vit_attn_tgt"], "viridis"),
    ("PyTorch", "MAE reconstruction", data["mae_rec_src"], None),
    ("Jittor", "MAE reconstruction", data["mae_rec_tgt"], None),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.titlepad": 5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})
fig, axes = plt.subplots(2, 2, figsize=(4.5, 4.15))
for ax, (title, label, arr, cmap) in zip(axes.flat, panels):
    if cmap is None:
        im = ax.imshow(arr, aspect="auto")
    else:
        im = ax.imshow(arr, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(label, labelpad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.tight_layout()
fig.savefig("plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
