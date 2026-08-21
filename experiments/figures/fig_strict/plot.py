from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

rows = list(csv.DictReader((Path(__file__).resolve().parent / "strict.csv").open()))
order = ["fp32 only", "strict pass"]
values = {row["kind"]: int(row["count"]) for row in rows}
labels = order
counts = [values.get(item, 0) for item in order]
colors = ["#A0CBE8", "#59A14F"]

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
fig, ax = plt.subplots(figsize=(3.45, 2.35))
bars = ax.bar(labels, counts, color=colors, width=0.32)
ax.set_ylabel("Cases", fontsize=8)
ax.set_title("FP32 certificate vs operator-strict parity")
ax.set_ylim(0, max(counts) + 1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.65)
ax.set_axisbelow(True)
ax.tick_params(direction="out", length=3)
for bar, value in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08, str(value), ha="center", fontsize=8)
fig.tight_layout()
fig.savefig("plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
