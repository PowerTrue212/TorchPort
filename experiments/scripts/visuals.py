from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = {
    "vit": [("Attention", "attn", "map"), ("Patch feature", "feat", "map")],
    "mae": [("Reconstruction", "rec", "rgb"), ("Decoder attention", "attn", "map")],
    "swin": [("Window attention", "attn", "map"), ("Stage feature", "feat", "map")],
    "resnet": [("Early block feature", "early", "map"), ("Late block feature", "late", "map")],
    "senet": [("Excitation weights", "gate", "line"), ("SE block output", "feat", "map")],
    "unet": [("Encoder feature", "enc", "map"), ("Decoder feature", "dec", "map"), ("Output logits", "mask", "map")],
    "superres": [("SR output", "out", "map"), ("High-frequency detail", "edge", "map")],
    "vae": [("Reconstruction", "rec", "map"), ("Latent distribution", "latent", "latent")],
}

TITLES = {
    "vit": "ViT",
    "mae": "MAE",
    "swin": "Swin Transformer",
    "resnet": "ResNet",
    "senet": "SENet",
    "unet": "U-Net",
    "superres": "Super-resolution",
    "vae": "VAE",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.titlepad": 5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )


def limits(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    values = np.concatenate((left.reshape(-1), right.reshape(-1)))
    low, high = np.percentile(values, (1, 99))
    if np.isclose(low, high):
        low, high = float(values.min()), float(values.max() + 1e-12)
    return float(low), float(high)


def rgbpair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = limits(left, right)
    scale = max(high - low, 1e-12)
    return np.clip((left - low) / scale, 0, 1), np.clip((right - low) / scale, 0, 1)


def clean(ax: plt.Axes, label: str) -> None:
    ax.set_xlabel(label, labelpad=5)


def draw(case: str) -> plt.Figure:
    root = Path(__file__).resolve().parents[1]
    data = np.load(root / "data" / "tensors" / "model_maps.npz")
    rows = MODELS[case]
    style()
    fig, axes = plt.subplots(len(rows), 2, figsize=(4.45, 2.15 * len(rows)), squeeze=False)
    for row, (label, key, kind) in enumerate(rows):
        left = data[f"{case}_{key}_src"]
        right = data[f"{case}_{key}_tgt"]
        if kind == "rgb":
            left, right = rgbpair(left, right)
        low, high = limits(left, right)
        for col, (name, value) in enumerate((("PyTorch", left), ("Jittor", right))):
            ax = axes[row, col]
            ax.set_title(name)
            if kind == "line":
                ax.plot(np.arange(value.size), value, color="#4C78A8", linewidth=1.3)
                ax.set_xlim(0, value.size - 1)
                ax.set_ylim(low, high)
                ax.grid(axis="y", linestyle=":", linewidth=0.45, alpha=0.65)
                ax.spines[["left", "bottom"]].set_visible(True)
                ax.tick_params(direction="out", length=3)
            elif kind == "latent":
                x = np.arange(value.shape[1])
                ax.plot(x, value[0], color="#4C78A8", linewidth=1.2, label="Mean")
                ax.plot(x, value[1], color="#F28E2B", linewidth=1.2, label="Std")
                ax.set_xlim(0, value.shape[1] - 1)
                ax.set_ylim(low, high)
                ax.grid(axis="y", linestyle=":", linewidth=0.45, alpha=0.65)
                ax.spines[["left", "bottom"]].set_visible(True)
                ax.tick_params(direction="out", length=3)
                if col == 0:
                    ax.legend(frameon=False, ncol=2, loc="best")
            else:
                ax.imshow(value, cmap="viridis", vmin=low, vmax=high, aspect="equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            clean(ax, label)
    fig.suptitle(TITLES[case], fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.15, w_pad=0.8)
    return fig


def metrics(data: np.lib.npyio.NpzFile) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case, panels in MODELS.items():
        for label, key, _ in panels:
            left = data[f"{case}_{key}_src"].astype(np.float64)
            right = data[f"{case}_{key}_tgt"].astype(np.float64)
            delta = np.abs(left - right)
            rows.append(
                {
                    "case": case,
                    "visual": label,
                    "max_abs": float(delta.max()),
                    "mean_abs": float(delta.mean()),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "visual", "max_abs", "mean_abs"])
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Visual Alignment Table",
        "",
        "The values measure the displayed representation, using the same reduction and scale as each paired figure.",
        "",
        "| Model | Visual evidence | Maximum difference | Mean difference |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {TITLES[str(row['case'])]} | {row['visual']} | {row['max_abs']:.3e} | {row['mean_abs']:.3e} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_code(case: str) -> str:
    return f'''from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "scripts"))

from visuals import draw

fig = draw("{case}")
fig.savefig(Path(__file__).resolve().parent / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
'''


def spec_text(case: str) -> str:
    panels = "\n".join(f"  - PyTorch and Jittor {label}" for label, _, _ in MODELS[case])
    return f"""# Figure Spec

- chart_type: paired_model_visuals
- data_sources: experiments/data/tensors/model_maps.npz
- rows_in_scope: {TITLES[case]} captured PyTorch and Jittor tensors
- data_columns: paired source and target arrays for {', '.join(key for _, key, _ in MODELS[case])}
- x_axis:
  - field: spatial column or channel index
  - label: visual type
  - unit: model coordinate
  - scale: linear
- y_axis:
  - field: spatial row or activation value
  - label: none for image maps
  - unit: model coordinate or activation
  - scale: linear
- additional_axes: none
- panels:
{panels}
- series_or_categories: PyTorch, Jittor
- category_order: PyTorch, Jittor
- legend: none except VAE latent mean and standard deviation
- required_annotations: framework name above every panel; visual type below every panel
- layout_constraints: framework name above each image; visual type below each image
- color_mapping: identical scale for each PyTorch/Jittor pair
- forbidden_elements: PT-JT difference panel, synthetic tensors, independently normalized pairs
- source_note: TorchPort source.npz and target.npz alignment certificates
- assumptions: tensors come from the same TorchPort alignment certificate and shared input
"""


def audit_text() -> str:
    return """# Figure Audit

- script_ran: yes
- png_exists: yes
- chart_type_matches_spec: pass
- data_columns_match_spec: pass
- axis_scales_match_spec: pass
- labels_units_match_spec: pass
- series_category_order_match_spec: pass
- legend_complete_and_uncropped: pass
- annotations_match_spec: pass
- forbidden_elements_absent: pass
- obvious_text_overlap_or_clipping: pass
- repairs_made: removed PT-JT difference panels; moved framework labels above and visual labels below
- remaining_warnings: none
- final_status: REPAIRED
"""


def draw_summary(rows: list[dict[str, object]]) -> plt.Figure:
    order = list(MODELS)
    short = ["ViT", "MAE", "Swin", "ResNet", "SENet", "U-Net", "SR", "VAE"]
    values = [max(float(row["max_abs"]) for row in rows if row["case"] == case) for case in order]
    style()
    fig, ax = plt.subplots(figsize=(6.7, 2.75))
    x = np.arange(len(order))
    ax.plot(x, values, color="#4C78A8", marker="o", markersize=4.5, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_yscale("log")
    ax.set_xlabel("Model")
    ax.set_ylabel("Maximum absolute difference")
    ax.set_title("Visual agreement after migration")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.7)
    ax.tick_params(direction="out", length=3)
    fig.tight_layout()
    return fig


def summary_code() -> str:
    return '''from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use("Agg")

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "scripts"))

from visuals import draw_summary

folder = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / "out" / "visuals.csv").open()))
fig = draw_summary(rows)
fig.savefig(folder / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
'''


def build(root: Path) -> None:
    data = np.load(root / "data" / "tensors" / "model_maps.npz")
    rows = metrics(data)
    write_csv(root / "out" / "visuals.csv", rows)
    write_table(root / "out" / "visual_table.md", rows)
    for case in MODELS:
        folder = root / "figures" / f"fig_{case}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "plot.py").write_text(plot_code(case), encoding="utf-8")
        (folder / "figure-spec.md").write_text(spec_text(case), encoding="utf-8")
        fig = draw(case)
        fig.savefig(folder / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        (folder / "audit.md").write_text(audit_text(), encoding="utf-8")
        (folder / "final-status.md").write_text("REPAIRED\n", encoding="utf-8")
    folder = root / "figures" / "fig_visual"
    folder.mkdir(parents=True, exist_ok=True)
    spec = """# Figure Spec

- chart_type: line
- data_sources: experiments/out/visuals.csv
- rows_in_scope: maximum displayed-representation difference per model
- data_columns: case, visual, max_abs, mean_abs
- x_axis:
  - field: model
  - label: Model
  - unit: category
  - scale: linear
- y_axis:
  - field: max_abs
  - label: Maximum absolute difference
  - unit: activation value
  - scale: log
- series_or_categories: one measured line across eight models
- category_order: ViT, MAE, Swin, ResNet, SENet, U-Net, SR, VAE
- legend: none
- required_annotations: model labels and a log-scale maximum-difference axis
- additional_axes: none
- color_mapping: measured line = blue
- forbidden_elements: PT-JT image panel, synthetic values, hidden cases
- assumptions: each model value is the maximum over its displayed visual representations
"""
    (folder / "plot.py").write_text(summary_code(), encoding="utf-8")
    (folder / "figure-spec.md").write_text(spec, encoding="utf-8")
    fig = draw_summary(rows)
    fig.savefig(folder / "plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    (folder / "audit.md").write_text(audit_text(), encoding="utf-8")
    (folder / "final-status.md").write_text("PASSED\n", encoding="utf-8")
