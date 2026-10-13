from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
COLORS = {
    "Codex": "#4C78A8",
    "DSH": "#E45756",
    "DSH + TorchPort": "#72A35A",
}


def number(value: str) -> float:
    return float(value) if value else np.nan


def load() -> list[dict[str, object]]:
    rows = []
    with (ROOT / "data.csv").open(encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    **raw,
                    "success": raw["success"].lower() == "true",
                    "rn": raw["rn"].lower() == "true",
                    "na": raw["na"].lower() == "true",
                    "rounds": int(raw["rounds"]),
                    **{
                        key: number(raw[key])
                        for key in (
                            "max_error",
                            "mean_error",
                            "mean_framework_flow_difference",
                            "p99_framework_flow_difference",
                            "max_framework_flow_difference",
                            "evaluator_seconds",
                            "agent_elapsed_seconds",
                            "non_cache_tokens",
                            "cache_read_tokens",
                        )
                    },
                }
            )
    return rows


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


def plot_comparison(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.15), gridspec_kw={"width_ratios": [0.9, 1.15]})

    ax = axes[0]
    x = np.arange(2)
    offsets = {"Codex": 0.04, "DSH": 0.0, "DSH + TorchPort": -0.04}
    for row in rows:
        name = str(row["display_name"])
        y = np.array([float(bool(row["rn"])), float(bool(row["na"]))]) + offsets[name]
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=5.5, color=COLORS[name], label=name)
    ax.set_xticks(x, ["RN", "NA"])
    ax.set_yticks([0, 1], ["Fail", "Pass"])
    ax.set_ylim(-0.14, 1.16)
    ax.set_xlim(-0.15, 1.15)
    ax.set_title("(a) Common evaluator")
    ax.set_ylabel("Outcome")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    metrics = [
        ("Mean", "mean_framework_flow_difference"),
        ("P99", "p99_framework_flow_difference"),
        ("Max", "max_framework_flow_difference"),
    ]
    for row in rows:
        name = str(row["display_name"])
        values = [float(row[key]) for _, key in metrics]
        ax.plot(range(3), values, marker="o", linewidth=1.7, markersize=5, color=COLORS[name])
        label_offset = {"Codex": (6, -9), "DSH": (6, 3), "DSH + TorchPort": (6, 9)}[name]
        ax.annotate(name, (2, values[-1]), xytext=label_offset, textcoords="offset points", va="center", fontsize=7.3)
    ax.set_xticks(range(3), [label for label, _ in metrics])
    ax.set_yscale("log")
    ax.set_ylabel("Framework flow difference")
    ax.set_title("(b) PyTorch–Jittor difference")

    for ax in axes:
        ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=3)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(ROOT / "comparison.png", dpi=300, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)


def plot_method_results(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(5.9, 3.0))
    label_offsets = {"Codex": (7, 7), "DSH + TorchPort": (7, 7)}
    for row in rows:
        name = str(row["display_name"])
        if name == "DSH":
            continue
        rounds = int(row["rounds"])
        passed = bool(row["success"])
        color = COLORS[name]
        if name == "DSH + TorchPort":
            x_values = np.array([0, rounds])
            stages = np.array([0, 1], dtype=float)
        elif passed and rounds > 1:
            x_values = np.arange(0, rounds + 1)
            stages = np.array([0] * rounds + [1], dtype=float)
        elif passed:
            x_values = np.array([0, rounds])
            stages = np.array([0, 1], dtype=float)
        else:
            x_values = np.arange(0, rounds + 1)
            stages = np.zeros(rounds + 1, dtype=float)
        marker = "s" if name == "DSH + TorchPort" else "o"
        marker_size = 4.8 if name == "DSH + TorchPort" else 7.0
        ax.plot(x_values, stages, color=color, marker=marker, markersize=marker_size, linewidth=1.8)
        if not passed:
            ax.scatter(x_values[-1], stages[-1], s=95, facecolor="white", edgecolor=color, linewidth=2.2, marker="X", zorder=4)
        label = name if passed else f"{name} · FAIL"
        ax.annotate(label, (x_values[-1], stages[-1]), xytext=label_offsets[name], textcoords="offset points", fontsize=8.0)
    ax.set_xticks(np.arange(0, 6))
    ax.set_xlim(-0.15, 5.75)
    ax.set_yticks([0, 1], ["RN", "NA"])
    ax.set_ylim(-0.24, 1.24)
    ax.set_xlabel("Repair round")
    ax.set_ylabel("Highest stage reached")
    ax.set_title("RAFT repair trajectories")
    ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3)
    fig.tight_layout()
    fig.savefig(ROOT / "method_results.png", dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


def paired_limits(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    values = np.concatenate((left.reshape(-1), right.reshape(-1)))
    low, high = np.percentile(values, (1, 99))
    return float(low), float(high)


def plot_features() -> None:
    data_path = ROOT.parents[1] / "data" / "tensors" / "raft_features.npz"
    with np.load(data_path, allow_pickle=False) as data:
        fig, axes = plt.subplots(2, 2, figsize=(5.7, 4.25), constrained_layout=True)
        for row in range(2):
            source = data[f"feature{row}_pytorch_rms"]
            target = data[f"feature{row}_jittor_rms"]
            low, high = paired_limits(source, target)
            images = []
            for column, (title, value) in enumerate((("PyTorch", source), ("Jittor", target))):
                axis = axes[row, column]
                images.append(axis.imshow(value, cmap="viridis", vmin=low, vmax=high, aspect="auto"))
                if row == 0:
                    axis.set_title(title)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
            axes[row, 0].set_ylabel(f"Feature {row}\nchannel RMS", labelpad=9)
            fig.colorbar(images[-1], ax=axes[row, :], fraction=0.025, pad=0.025)
        fig.suptitle("RAFT intermediate feature magnitude", fontsize=11)
        fig.savefig(ROOT / "feature_pairs.png", dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)

        errors = [data[f"feature{index}_error_rms"] for index in range(2)]
        maximum = max(float(value.max()) for value in errors)
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.45), constrained_layout=True)
        image = None
        for index, (axis, value) in enumerate(zip(axes, errors)):
            image = axis.imshow(value, cmap="magma", vmin=0.0, vmax=maximum, aspect="auto")
            axis.set_title(f"Feature {index}")
            axis.set_xlabel(f"max RMS difference {float(value.max()):.2e}")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        if image is not None:
            fig.colorbar(image, ax=axes, fraction=0.026, pad=0.025, label="Channel RMS difference")
        fig.suptitle("Intermediate feature difference: Jittor vs PyTorch", fontsize=11)
        fig.savefig(ROOT / "feature_errors.png", dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)


def plot_iteration_convergence() -> None:
    with (ROOT / "iteration_metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    iterations = np.array([int(row["iteration"]) for row in rows])
    source_update = np.array([float(row["pytorch_update_mean"]) for row in rows])
    target_update = np.array([float(row["jittor_update_mean"]) for row in rows])
    mean_difference = np.array([float(row["framework_mean"]) for row in rows])
    p99_difference = np.array([float(row["framework_p99"]) for row in rows])
    max_difference = np.array([float(row["framework_max"]) for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    axes[0].plot(iterations, source_update, color="#4C78A8", marker="o", markersize=4, linewidth=1.6, label="PyTorch")
    axes[0].plot(iterations, target_update, color="#72A35A", marker="s", markersize=3.5, linewidth=1.4, linestyle="--", label="Jittor")
    axes[0].set_title("(a) RAFT update magnitude")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Mean endpoint update")
    axes[0].legend(frameon=False)

    axes[1].plot(iterations, mean_difference, color="#4C78A8", marker="o", markersize=3.5, linewidth=1.5, label="Mean")
    axes[1].plot(iterations, p99_difference, color="#F2B134", marker="s", markersize=3.2, linewidth=1.5, label="P99")
    axes[1].plot(iterations, max_difference, color="#E45756", marker="^", markersize=3.8, linewidth=1.5, label="Max")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Framework flow difference")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Endpoint difference")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(iterations)
        axis.grid(True, linestyle=":", linewidth=0.55, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", length=3)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(ROOT / "iteration_convergence.png", dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def plot_strict_profile() -> None:
    with (ROOT / "tensor_alignment.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    flow_up = {int(row["name"][-2:]): 100.0 * float(row["strict_mismatch_ratio"]) for row in rows if row["kind"] == "flow_up"}
    flow_low = {int(row["name"][-2:]): 100.0 * float(row["strict_mismatch_ratio"]) for row in rows if row["kind"] == "flow_low"}
    features = [row for row in rows if row["kind"] == "feature"]
    iterations = np.arange(1, 13)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.35, 0.65]})
    axes[0].plot(iterations, [flow_up[index] for index in iterations], color="#4C78A8", marker="o", markersize=4, linewidth=1.6, label="flow_up")
    axes[0].plot(iterations, [flow_low[index] for index in iterations], color="#E45756", marker="s", markersize=3.6, linewidth=1.6, label="flow_low")
    axes[0].set_xticks(iterations)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Strict mismatched elements (%)")
    axes[0].set_title("(a) Strict mismatch accumulation")
    axes[0].legend(frameon=False)

    feature_names = [row["name"] for row in features]
    feature_values = [100.0 * float(row["strict_mismatch_ratio"]) for row in features]
    bars = axes[1].bar(feature_names, feature_values, color=["#72A35A", "#9C78D4"], width=0.48)
    for bar, value in zip(bars, feature_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}%", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_ylabel("Strict mismatch (%)")
    axes[1].set_title("(b) Encoder features")
    for axis in axes:
        axis.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", length=3)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(ROOT / "strict_profile.png", dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def main() -> None:
    style()
    rows = load()
    plot_comparison(rows)
    plot_method_results(rows)
    plot_features()
    plot_iteration_convergence()
    plot_strict_profile()


if __name__ == "__main__":
    main()
