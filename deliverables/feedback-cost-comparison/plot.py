from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter


ROOT = Path(__file__).resolve().parent
NONE_COLOR = "#69AEEB"
FULL_COLOR = "#F5A532"
EDGE_COLOR = "#202020"
GRID_COLOR = "#D0D0D0"

# Cumulative run statistics used in the experiment summary.
COST_DATA = {
    "PlantCamo": {
        "Token": {"none": 38.663183, "full": 9.271686},
        "Logical queries": {"none": 10, "full": 1},
        "Steps": {"none": 236, "full": 95},
    },
    "VSCode-v2-T": {
        "Token": {"none": 74.328976, "full": 79.488352},
        "Logical queries": {"none": 15, "full": 5},
        "Steps": {"none": 325, "full": 390},
    },
}

RAW_LABELS = {
    "PlantCamo": {
        "Token": {"none": "38.7M", "full": "9.27M"},
        "Logical queries": {"none": "10", "full": "1"},
        "Steps": {"none": "236", "full": "95"},
    },
    "VSCode-v2-T": {
        "Token": {"none": "74.3M", "full": "79.5M"},
        "Logical queries": {"none": "15", "full": "5"},
        "Steps": {"none": "325", "full": "390"},
    },
}

E2_MAX_DIFF = {
    "CamoFormer": 1.0e-7,
    "BiRefNet": 8.5e-8,
    "PlantCamo": 5.6e-8,
    "VSCode-v2-T": 2.75e-7,
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "SimHei",
                "DejaVu Sans",
            ],
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.unicode_minus": False,
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.grid(axis="y", linestyle="--", linewidth=0.65, color=GRID_COLOR, alpha=0.8, zorder=0)
    for spine in axis.spines.values():
        spine.set_color(EDGE_COLOR)
        spine.set_linewidth(1.15)


def add_common_legend(figure: plt.Figure) -> None:
    figure.legend(
        handles=[
            Patch(facecolor=NONE_COLOR, edgecolor=EDGE_COLOR, label="none"),
            Patch(facecolor=FULL_COLOR, edgecolor=EDGE_COLOR, label="full"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        frameon=True,
        facecolor="white",
        edgecolor=EDGE_COLOR,
        framealpha=1.0,
        ncol=2,
        columnspacing=1.5,
    )


def add_value_labels(
    axis: plt.Axes,
    bars,
    labels: list[str],
    offset: float,
    skip: set[int] | None = None,
) -> None:
    skip = skip or set()
    for index, (bar, label) in enumerate(zip(bars, labels)):
        if index in skip:
            continue
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9.0,
        )


def plot_e2_consistency() -> None:
    repositories = list(E2_MAX_DIFF)
    values = [E2_MAX_DIFF[name] for name in repositories]
    value_labels = [
        r"$\approx 1.0\times10^{-7}$",
        r"$8.5\times10^{-8}$",
        r"$5.6\times10^{-8}$",
        r"$2.75\times10^{-7}$",
    ]
    # Keep the canvas size unchanged, but bring the four categories closer
    # together inside the plotting area.
    x = [0.0, 0.7, 1.4, 2.1]

    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    figure.patch.set_facecolor("white")
    style_axis(axis)
    axis.set_yscale("log")

    axis.scatter(
        x,
        values,
        s=90,
        marker="o",
        color=NONE_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.15,
        zorder=3,
    )
    for position, value, label in zip(x, values, value_labels):
        axis.annotate(
            label,
            (position, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    axis.set_xlim(-0.75, 2.85)
    axis.set_ylim(4.0e-8, 3.8e-7)
    axis.set_yticks(
        [5.0e-8, 1.0e-7, 2.0e-7, 3.0e-7],
        [r"$5\times10^{-8}$", r"$10^{-7}$", r"$2\times10^{-7}$", r"$3\times10^{-7}$"],
    )
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xticks(x, repositories)
    axis.set_ylabel("E2 最大指标差")
    figure.subplots_adjust(left=0.19, right=0.98, top=0.96, bottom=0.18)

    figure.savefig(ROOT / "fig1_e2_consistency.png", dpi=300, facecolor="white")
    figure.savefig(ROOT / "fig1_e2_consistency.pdf", facecolor="white")
    plt.close(figure)


def normalized_values(repository: str, metric: str) -> tuple[float, float]:
    values = COST_DATA[repository][metric]
    maximum = max(values.values())
    return values["none"] / maximum * 100, values["full"] / maximum * 100


def plot_costs() -> None:
    repositories = ["PlantCamo", "VSCode-v2-T"]
    metrics = ["Token", "Logical queries", "Steps"]
    tick_labels = ["Token", "逻辑查询", "步数", "Token", "逻辑查询", "步数"]
    x = [0.0, 0.9, 1.8, 3.05, 3.95, 4.85]
    width = 0.34

    none_heights: list[float] = []
    full_heights: list[float] = []
    none_labels: list[str] = []
    full_labels: list[str] = []
    for repository in repositories:
        for metric in metrics:
            none_value, full_value = normalized_values(repository, metric)
            none_heights.append(none_value)
            full_heights.append(full_value)
            none_labels.append(RAW_LABELS[repository][metric]["none"])
            full_labels.append(RAW_LABELS[repository][metric]["full"])

    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    figure.patch.set_facecolor("white")
    style_axis(axis)

    none_bars = axis.bar(
        [position - width / 2 for position in x],
        none_heights,
        width,
        color=NONE_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.05,
        zorder=3,
    )
    full_bars = axis.bar(
        [position + width / 2 for position in x],
        full_heights,
        width,
        color=FULL_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.05,
        zorder=3,
    )

    add_value_labels(axis, none_bars, none_labels, 2.0)
    add_value_labels(axis, full_bars, full_labels, 2.0)

    axis.set_xticks(x, tick_labels)
    axis.set_xlim(-0.55, 5.4)
    axis.set_ylim(0, 114)
    axis.set_yticks([0, 20, 40, 60, 80, 100])
    axis.set_ylabel("相对成本 / %")
    axis.axvline(2.43, color="#B9B9B9", linestyle=":", linewidth=1.0, zorder=1)

    axis.text(
        0.9,
        -0.18,
        "PlantCamo",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )
    axis.text(
        3.95,
        -0.18,
        "VSCode-v2-T",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )

    add_common_legend(figure)
    figure.subplots_adjust(left=0.11, right=0.985, top=0.84, bottom=0.25)

    figure.savefig(ROOT / "fig2_cost_comparison.png", dpi=300, facecolor="white")
    figure.savefig(ROOT / "fig2_cost_comparison.pdf", facecolor="white")
    # Keep the earlier filename pointing at the revised compact Figure 2.
    figure.savefig(ROOT / "feedback_cost_comparison.png", dpi=300, facecolor="white")
    figure.savefig(ROOT / "feedback_cost_comparison.pdf", facecolor="white")
    plt.close(figure)


def main() -> None:
    style()
    plot_e2_consistency()
    plot_costs()


if __name__ == "__main__":
    main()
