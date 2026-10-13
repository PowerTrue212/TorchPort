from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
COLORS = {
    "Codex": "#4C78A8",
    "DSH": "#E45756",
    "DSH + TorchPort": "#72A35A",
    "DSH V4 Flash + TP": "#8B6BB1",
}
MARKERS = {
    "Codex": "o",
    "DSH": "s",
    "DSH + TorchPort": "^",
    "DSH V4 Flash + TP": "D",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
        }
    )


def plot_method_results() -> None:
    rows = read_csv("method_results.csv")
    headers = ["Task", "Method", "RN", "NA", "Rounds"]
    values = [
        [
            row["model"],
            row["method"].replace("DSH + TorchPort", "DSH + TP"),
            "PASS" if row["rn"] == "true" else "FAIL",
            "PASS" if row["na"] == "true" else "FAIL",
            row["rounds"],
        ]
        for row in rows
    ]
    figure, axis = plt.subplots(figsize=(7.15, 3.05))
    axis.axis("off")
    table = axis.table(
        cellText=values,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.12, 0.43, 0.11, 0.11, 0.13],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.42)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.55)
        if row_index == 0:
            cell.set_facecolor("#E9EDF2")
            cell.set_text_props(weight="bold", color="#222222")
            continue
        cell.set_facecolor("#FAFAFA" if row_index % 2 else "#F3F5F7")
        if column_index in (2, 3):
            passed = cell.get_text().get_text() == "PASS"
            cell.set_facecolor("#E5F2E1" if passed else "#F8DFDE")
            cell.set_text_props(color="#2F6B2F" if passed else "#A33A38", weight="bold")
        if column_index == 1 and "V4 Flash" in cell.get_text().get_text():
            cell.set_facecolor("#EEE8F5")
            cell.set_text_props(weight="bold", color="#4D3C64")
    axis.set_title("Common-evaluator outcomes", pad=8)
    figure.tight_layout(pad=0.35)
    figure.savefig(ROOT / "method_results.png", dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(figure)


def plot_diagnosis_quality() -> None:
    rows = read_csv("diagnosis_quality.csv")
    methods = [row["method"] for row in rows]
    values = [int(row["strict_mismatches"]) for row in rows]
    colors = [COLORS[method] for method in methods]
    labels = [method.replace("DSH + TorchPort", "DSH + TP") for method in methods]

    figure, axis = plt.subplots(figsize=(5.9, 2.65))
    bars = axis.barh(labels, values, color=colors, height=0.48, alpha=0.9)
    axis.invert_yaxis()
    axis.set_xlim(0, max(values) * 1.23)
    axis.set_xlabel("Strict mismatches at feature.0  (lower is better)")
    axis.set_title("Diagnosis quality at the first divergence")
    axis.grid(axis="x", linestyle=":", linewidth=0.55, alpha=0.6)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    for bar, value in zip(bars, values):
        axis.text(
            value + max(values) * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            ha="left",
            fontsize=9,
            weight="bold",
            color="#222222",
        )
    improvement = values[0] / values[1]
    axis.text(
        0.98,
        0.08,
        f"{improvement:.1f}× fewer mismatches",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color=COLORS["DSH + TorchPort"],
        fontsize=9,
        weight="bold",
    )
    figure.tight_layout()
    figure.savefig(ROOT / "diagnosis_quality.png", dpi=300, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(figure)


def plot_agent_costs() -> None:
    rows = [row for row in read_csv("agent_costs.csv") if row["model"] == "RAFT"]
    method_order = ["Codex", "DSH", "DSH + TorchPort", "DSH V4 Flash + TP"]
    rows.sort(key=lambda row: method_order.index(row["method"]))
    size_scale = 1.0 / 280.0
    figure, axis = plt.subplots(figsize=(6.7, 3.45))
    for x, row in enumerate(rows):
        method = row["method"]
        hours = float(row["active_seconds"]) / 3600.0
        tokens = float(row["non_cache_tokens"])
        passed = row["status"] == "passed"
        bubble_size = tokens * size_scale
        axis.scatter(
            x,
            hours,
            s=bubble_size,
            color=COLORS[method],
            edgecolor=COLORS[method],
            linewidth=0.8,
            marker="o",
            alpha=0.88,
            zorder=3,
        )
        status_line = "\nFAILED" if not passed else ""
        axis.text(
            x,
            hours,
            f"{hours:.2f} h\n{tokens / 1_000_000:.2f}M{status_line}",
            ha="center",
            va="center",
            fontsize=7.2,
            weight="bold",
            color="white",
            zorder=5,
        )
    tick_labels = ["Codex", "DSH", "DSH + TP", "V4 Flash + TP"]
    axis.set_xticks(range(len(rows)), tick_labels)
    axis.set_xlim(-0.5, len(rows) - 0.5)
    axis.set_ylim(0.25, 4.55)
    axis.set_ylabel("Active agent time (hours)")
    axis.set_title("RAFT agent cost")
    axis.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3)
    figure.text(0.5, 0.01, "Inside each bubble: active time and non-cache tokens   ·   bubble area ∝ tokens", ha="center", fontsize=7.2, color="#555555")
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(ROOT / "agent_cost_bubbles.png", dpi=300, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(figure)


def main() -> None:
    style()
    plot_method_results()
    plot_diagnosis_quality()
    plot_agent_costs()


if __name__ == "__main__":
    main()
