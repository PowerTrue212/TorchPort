from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import NullLocator

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
INK = "#171717"
MUTED = "#777777"
GRID = "#D8D8D8"
CORAL = "#D95364"
AMBER = "#E5A33A"
BLUE = "#4477AA"
TEAL = "#078A82"


def configure_style() -> None:
    font_path = Path(r"C:\Windows\Fonts\STSONG.TTF")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "STSong",
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.35,
            color=color,
            shrinkA=2,
            shrinkB=3,
        )
    )


def draw_mechanism(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    ys = [0.75, 0.52, 0.29]
    colors = [BLUE, AMBER, CORAL]
    for index, (y_value, color) in enumerate(zip(ys, colors), start=1):
        axis.text(0.08, y_value, f"T{index}", ha="center", va="center", color=INK)
        arrow(axis, (0.15, y_value), (0.49, 0.52), color)

    node = Circle((0.62, 0.52), 0.13, facecolor="#FFF4F1", edgecolor=CORAL, linewidth=1.45)
    axis.add_patch(node)
    axis.text(0.62, 0.52, "atomicAdd", ha="center", va="center", fontsize=9)
    arrow(axis, (0.76, 0.52), (0.93, 0.52), INK)
    axis.text(0.97, 0.52, r"$\Sigma$", ha="center", va="center", fontsize=16)
    axis.text(
        0.52,
        0.10,
        r"$(a+b)+c\;\neq\;a+(b+c)$",
        ha="center",
        va="center",
        fontsize=11,
        color=INK,
    )


def draw_measurements(axis: plt.Axes) -> None:
    results = json.loads(
        (REPO / "docs" / "evidence" / "jittor_atomic_reduction" / "results.json").read_text(
            encoding="utf-8"
        )
    )["metrics"]
    pool_sum = max(
        results["builtin_sum_spatial"]["max_relative_l2_run_diff"],
        results["builtin_avgpool"]["max_relative_l2_run_diff"],
    )
    values = np.array(
        [
            pool_sum,
            results["explicit_atomic"]["max_relative_l2_run_diff"],
            results["conv3x3"]["max_relative_l2_run_diff"],
            results["fixed_tree"]["max_relative_l2_run_diff"],
        ]
    )
    labels = ["Pool/Sum", "原子加法", "Conv", "固定加法树"]
    colors = [AMBER, CORAL, TEAL, BLUE]
    positions = np.arange(3, -1, -1)

    axis.scatter(
        values,
        positions,
        s=72,
        color=colors,
        edgecolor=INK,
        linewidth=0.8,
        zorder=3,
    )
    annotations = [
        r"$2.0\times10^{-6}$",
        r"$1.4\times10^{-6}$",
        r"$1.8\times10^{-7}$",
        r"$0$",
    ]
    offsets = [
        (-8, 0, "right"),
        (-8, 0, "right"),
        (8, 0, "left"),
        (-8, 0, "right"),
    ]
    for x_value, y_value, label, (x_offset, y_offset, alignment) in zip(
        values, positions, annotations, offsets
    ):
        axis.annotate(
            label,
            (x_value, y_value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=8.5,
            color=INK,
        )

    axis.set_xscale("symlog", linthresh=1e-10, linscale=0.6, base=10)
    axis.set_xlim(-1.2e-10, 1.5e-5)
    axis.set_ylim(-0.48, 3.48)
    axis.set_yticks(positions, labels)
    axis.set_xticks([0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
    axis.set_xlabel("Run-to-run relative difference")
    axis.xaxis.set_minor_locator(NullLocator())
    axis.grid(axis="x", color=GRID, linewidth=0.7, linestyle=(0, (4, 4)))
    axis.set_axisbelow(True)
    axis.tick_params(direction="in", top=True, right=True, width=0.85, length=3.8)
    for spine in axis.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(1.0)


def main() -> None:
    configure_style()
    figure = plt.figure(figsize=(7.4, 3.15))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.28)
    draw_measurements(figure.add_subplot(grid[0]))
    draw_mechanism(figure.add_subplot(grid[1]))
    figure.subplots_adjust(left=0.12, right=0.985, top=0.96, bottom=0.22)
    figure.savefig(ROOT / "jittor_atomic_mechanism.png", dpi=360, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(ROOT / "jittor_atomic_mechanism.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


if __name__ == "__main__":
    main()
