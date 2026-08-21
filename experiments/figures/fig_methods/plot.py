from __future__ import annotations

import csv
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

rows = [
    row
    for row in csv.DictReader((Path(__file__).resolve().parent / "methods.csv").open())
    if int(row["cases"]) > 0
]
by_method = {row["method"]: row for row in rows}
metrics = [
    ("RN", "verified"),
    ("NA", "fp32"),
    ("SA", "strict"),
]
methods = [("official_jittor", "Official", "#9C755F"), ("dsh_torchport", "Ours", "#4C78A8")]
y = np.arange(len(metrics))
height = 0.24

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titlepad": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.9,
})
fig, ax = plt.subplots(figsize=(4.75, 2.55))
for idx, (method, label, color) in enumerate(methods):
    row = by_method.get(method)
    if not row:
        continue
    total = max(int(row["cases"]), 1)
    rates = [100.0 * int(row[key]) / total for _, key in metrics]
    ypos = y + (0.5 - idx) * height
    bars = ax.barh(ypos, rates, height, label=label, color=color)
    for bar, rate in zip(bars, rates):
        x_text = 2.2 if rate == 0 else rate + 2.0 if rate < 92 else rate - 4.0
        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.0f}%",
            ha="left" if rate < 92 else "right",
            va="center",
            fontsize=7,
            color=color if rate < 92 else "white",
            fontweight="bold" if rate >= 92 else "normal",
        )

ax.set_xlabel("Success rate (%)", fontsize=8)
ax.set_title("End-to-end outcomes")
ax.set_yticks(y)
ax.set_yticklabels([name for name, _ in metrics])
ax.set_xlim(0, 108)
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle=":", linewidth=0.55, alpha=0.65)
ax.set_axisbelow(True)
ax.tick_params(direction="out", length=3)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
fig.tight_layout()
fig.savefig("plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
