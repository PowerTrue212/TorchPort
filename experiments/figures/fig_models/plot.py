from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

rows = list(csv.DictReader((Path(__file__).resolve().parent / "models.csv").open()))
case_order = ["vit_pytorch", "mae", "swin", "resnet_cifar10", "senet", "pytorch_unet", "superres", "vae"]
metric_order = ["RN", "NA", "SA"]
values = {(row["case"], row["metric"]): int(row["value"]) for row in rows}
matrix = np.array([[values.get((case, metric), 0) for metric in metric_order] for case in case_order])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titlepad": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.9,
})
fig, ax = plt.subplots(figsize=(4.5, 3.0))
cmap = ListedColormap(["#E6E6E6", "#4C78A8"])
ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(metric_order)))
ax.set_xticklabels(metric_order, rotation=0)
ax.set_yticks(range(len(case_order)))
ax.set_yticklabels(case_order)
ax.set_title("Model-level migration outcomes")
for y in range(matrix.shape[0]):
    for x in range(matrix.shape[1]):
        text = "✓" if matrix[y, x] else "–"
        color = "white" if matrix[y, x] else "#555555"
        ax.text(x, y, text, ha="center", va="center", fontsize=10, color=color, fontweight="bold")
ax.set_xticks(np.arange(-0.5, len(metric_order), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(case_order), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
ax.tick_params(which="minor", bottom=False, left=False)
ax.tick_params(direction="out", length=3)
fig.tight_layout()
fig.savefig("plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
