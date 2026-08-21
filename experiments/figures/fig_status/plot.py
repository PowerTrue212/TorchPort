from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
data = json.loads((root / "data" / "current.json").read_text(encoding="utf-8"))
rows = data["rows"]
labels = ["Verified", "FP32 parity", "Strict parity"]
values = [
    sum(1 for row in rows if row["status"] == "verified"),
    sum(1 for row in rows if row["fp32"]),
    sum(1 for row in rows if row["strict"]),
]
colors = ["#4C78A8", "#59A14F", "#F28E2B"]

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
fig, ax = plt.subplots(figsize=(3.7, 2.45))
bars = ax.bar(labels, values, color=colors, width=0.34)
ax.set_ylabel("Cases", fontsize=8)
ax.set_title("TorchPort current verified snapshot")
ax.set_ylim(0, max(values) + 1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.65)
ax.set_axisbelow(True)
ax.tick_params(direction="out", length=3)
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08, str(value), ha="center", fontsize=8)
fig.tight_layout()
fig.savefig("plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
