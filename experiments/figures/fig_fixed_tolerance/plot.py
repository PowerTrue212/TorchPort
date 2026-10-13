from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import NullLocator

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parent

INK = "#161616"
GRID = "#D7D7D7"
BLUE = "#4C78A8"
TEAL = "#0B8F87"
CORAL = "#D64D5F"
AMBER = "#E4A33B"


def configure_style() -> None:
    font_path = Path(r"C:\Windows\Fonts\STSONG.TTF")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)

    plt.rcParams.update(
        {
            "font.family": "STSong",
            "font.size": 9.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.05,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def style_axis(axis: plt.Axes, *, grid_axis: str = "both") -> None:
    axis.set_axisbelow(True)
    axis.grid(
        axis=grid_axis,
        which="major",
        color=GRID,
        linewidth=0.72,
        linestyle=(0, (4, 4)),
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(1.05)
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        width=0.9,
        length=4.2,
        color=INK,
    )
    axis.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        width=0.75,
        length=2.4,
        color=INK,
    )


def save(figure: plt.Figure, stem: str) -> None:
    figure.savefig(ROOT / f"{stem}.png", dpi=360, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(ROOT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)


def plot_fixed_tolerance() -> None:
    counts = np.array([75_973, 569])
    total = 81_920
    values = counts / total * 100
    positions = np.arange(2)

    figure = plt.figure(figsize=(4.45, 3.35))
    grid = figure.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0)
    upper = figure.add_subplot(grid[0])
    lower = figure.add_subplot(grid[1], sharex=upper)

    for axis in (upper, lower):
        axis.bar(
            positions,
            values,
            width=0.54,
            color=[CORAL, BLUE],
            edgecolor=INK,
            linewidth=1.25,
            zorder=3,
        )
        style_axis(axis, grid_axis="y")

    upper.set_ylim(84, 102)
    upper.set_yticks([90, 95, 100])
    lower.set_ylim(0, 2.0)
    lower.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    lower.set_xticks(positions, ["On", "Off"])
    upper.tick_params(labelbottom=False, bottom=False)
    lower.tick_params(top=False)
    upper.spines["bottom"].set_visible(False)
    lower.spines["top"].set_visible(False)

    upper.text(
        0.98,
        0.94,
        "Tolerance = 1×10⁻⁴",
        transform=upper.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        fontfamily="Times New Roman",
        color=INK,
    )
    upper.text(
        positions[0],
        values[0] + 0.7,
        f"{values[0]:.1f}%\n{counts[0]:,} / {total:,}",
        ha="center",
        va="bottom",
        fontsize=9.2,
        linespacing=1.15,
        color=INK,
    )
    lower.text(
        positions[1],
        values[1] + 0.12,
        f"{values[1]:.1f}%\n{counts[1]:,} / {total:,}",
        ha="center",
        va="bottom",
        fontsize=9.2,
        linespacing=1.15,
        color=INK,
    )

    marker_size = 0.014
    break_style = dict(color=INK, clip_on=False, linewidth=1.05)
    upper.plot(
        (-marker_size, +marker_size),
        (-marker_size, +marker_size),
        transform=upper.transAxes,
        **break_style,
    )
    upper.plot(
        (1 - marker_size, 1 + marker_size),
        (-marker_size, +marker_size),
        transform=upper.transAxes,
        **break_style,
    )
    lower.plot(
        (-marker_size, +marker_size),
        (1 - marker_size, 1 + marker_size),
        transform=lower.transAxes,
        **break_style,
    )
    lower.plot(
        (1 - marker_size, 1 + marker_size),
        (1 - marker_size, 1 + marker_size),
        transform=lower.transAxes,
        **break_style,
    )

    figure.supxlabel("Low-precision matmul (TF32)", fontsize=10.5, y=0.075)
    figure.supylabel("Outputs above tolerance (%)", fontsize=10.5, x=0.075)
    figure.subplots_adjust(left=0.19, right=0.97, top=0.96, bottom=0.20)
    save(figure, "01_fixed_tolerance")


def plot_runtime_variation() -> None:
    labels = ["Else", "Conv", "Pool/Sum"]
    values = np.array([0.0, 1.8e-7, 2.0e-6])
    positions = np.arange(3)
    colors = [BLUE, TEAL, AMBER]

    figure, axis = plt.subplots(figsize=(4.75, 3.35))
    axis.scatter(
        values,
        positions,
        s=98,
        color=colors,
        edgecolor=INK,
        linewidth=0.9,
        zorder=4,
    )
    annotations = ["0", r"$1.8\times10^{-7}$", r"$2.0\times10^{-6}$"]
    offsets = [(-7, 0, "right"), (7, 0, "left"), (-7, 0, "right")]
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
            fontsize=9.2,
            color=INK,
        )

    axis.set_xscale("symlog", linthresh=1e-10, linscale=0.6, base=10)
    axis.set_xlim(-1.2e-10, 1.5e-5)
    axis.set_ylim(-0.48, 2.48)
    axis.set_yticks(positions, labels)
    axis.set_xticks([0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
    axis.set_xlabel("Run-to-run relative difference")
    axis.xaxis.set_minor_locator(NullLocator())

    style_axis(axis, grid_axis="x")
    figure.subplots_adjust(left=0.35, right=0.97, top=0.96, bottom=0.20)
    save(figure, "02_runtime_variation")


def plot_error_amplification() -> None:
    evidence_path = ROOT.parents[2] / "docs" / "evidence" / "precision_small" / "results.json"
    results = json.loads(evidence_path.read_text(encoding="utf-8"))
    cells = results["B"]["cells"]
    high = np.asarray(cells["no_norm|random_too_large|real_data"]["gain_per_layer"])
    low = np.asarray(cells["no_norm|random_xavier|random_noise"]["gain_per_layer"])
    layers = np.arange(1, high.size + 1)

    figure, axis = plt.subplots(figsize=(4.75, 3.35))
    axis.plot(
        layers,
        high,
        color=CORAL,
        linewidth=2.25,
        marker="D",
        markersize=5.0,
        markeredgecolor=INK,
        markeredgewidth=0.7,
        label="Oversized init. + real data",
        zorder=3,
    )
    axis.plot(
        layers,
        low,
        color=TEAL,
        linewidth=2.05,
        linestyle=(0, (6, 3)),
        marker="^",
        markersize=5.5,
        markeredgecolor=INK,
        markeredgewidth=0.7,
        label="Xavier init. + random noise",
        zorder=3,
    )

    axis.text(
        0.97,
        0.50,
        "≈ 18 orders",
        transform=axis.transAxes,
        ha="right",
        va="center",
        fontsize=9.2,
        color=INK,
    )

    axis.set_yscale("log")
    axis.set_xlim(1, 15)
    axis.set_ylim(1e-5, 1e15)
    axis.set_xticks([1, 3, 6, 9, 12, 15])
    axis.set_xlabel("Layer")
    axis.set_ylabel("Estimated gain")
    axis.yaxis.set_minor_locator(NullLocator())
    axis.legend(loc="upper left", frameon=False, fontsize=8.2, handlelength=2.5)

    style_axis(axis)
    figure.subplots_adjust(left=0.18, right=0.97, top=0.96, bottom=0.20)
    save(figure, "03_error_amplification")


def main() -> None:
    configure_style()
    plot_fixed_tolerance()
    plot_runtime_variation()
    plot_error_amplification()


if __name__ == "__main__":
    main()
