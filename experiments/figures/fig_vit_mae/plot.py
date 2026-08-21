from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

rows = list(csv.DictReader((Path(__file__).resolve().parent / "vit_mae.csv").open()))
rows = sorted(rows, key=lambda row: 0 if row["case"] == "vit_pytorch" else 1)
labels = {"vit_pytorch": "ViT", "mae": "MAE"}

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
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5), sharey=True)
for ax, row in zip(axes, rows):
    case = row["case"]
    checks = [
        ("RN", row["status"] == "verified"),
        ("NA", row["fp32"] == "True"),
        ("SA", row["strict"] == "True"),
    ]
    colors = ["#4C78A8" if ok else "#D9D9D9" for _, ok in checks]
    ax.barh([name for name, _ in checks], [1, 1, 1], color=colors, height=0.45)
    for i, (_, ok) in enumerate(checks):
        ax.text(0.5, i, "pass" if ok else "not strict", ha="center", va="center", color="white" if ok else "#555555", fontsize=7, fontweight="bold" if ok else "normal")
    weights = f"{row['weights_loaded']}/{row['weights_total']} weights"
    first = row["first"] if row["first"] else "none"
    ax.text(0.0, -0.78, weights, fontsize=7, color="#333333")
    ax.text(0.0, -1.05, f"first strict divergence: {first}", fontsize=7, color="#333333")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title(labels.get(case, case))
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
fig.suptitle("Classic transformer cases", fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig("plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
