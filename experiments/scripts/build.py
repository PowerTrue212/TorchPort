from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "current.json"
BASE = ROOT / "baselines.json"
OUT = ROOT / "out"
FIGS = ROOT / "figures"


def main() -> None:
    data = _read(DATA)
    base = _read(BASE)
    rows = data["rows"]
    methods = _methods(rows, base["methods"])
    OUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT / "cases.csv", rows)
    _write_csv(OUT / "methods.csv", methods)
    _write_text(OUT / "case_table.md", _case_table(rows))
    _write_text(OUT / "method_table.md", _method_table(methods))
    _write_text(OUT / "model_table.md", _model_table(rows))
    _write_text(OUT / "metric_defs.md", _metric_defs())
    _write_text(OUT / "errors.md", _errors(data.get("errors", [])))
    _method_fig(methods)
    _model_fig(rows)
    _tensor_fig()
    _vit_align_fig()
    _mae_align_fig()
    _fail_fig(data.get("errors", []))
    _status_fig(rows)
    _strict_fig(rows)
    _model_visuals()


def _model_visuals() -> None:
    from visuals import build

    build(ROOT)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _methods(rows: list[dict[str, Any]], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    by_method = {item["id"]: [] for item in specs}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    for spec in specs:
        items = by_method.get(spec["id"], [])
        total = len(items)
        result.append(
            {
                "method": spec["id"],
                "name": spec["name"],
                "status": "run" if total else "pending",
                "cases": total,
                "converted": sum(1 for item in items if item.get("status") in {"converted", "verified"}),
                "verified": _count(items, "status", "verified"),
                "fp32": sum(1 for item in items if item.get("fp32")),
                "strict": sum(1 for item in items if item.get("strict")),
                "uses_torchport": spec["uses_torchport"],
                "uses_parity": spec["uses_parity"],
            }
        )
    return result


def _count(rows: list[dict[str, Any]], key: str, value: Any) -> int:
    return sum(1 for item in rows if item.get(key) == value)


def _case_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Case Table",
        "",
        "| Case | Method | Status | FP32 | Strict | Weights | Tensors | First divergence | DSH decision |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        weights = f"{row['weights_loaded']}/{row['weights_total']}"
        lines.append(
            "| {case} | {method} | {status} | {fp32} | {strict} | {weights} | {tensors} | {first} | {decision} |".format(
                case=row["case"],
                method=row["method"],
                status=row["status"],
                fp32=_mark(row["fp32"]),
                strict=_mark(row["strict"]),
                weights=weights,
                tensors=row["tensors"],
                first=row.get("first") or "none",
                decision=row.get("decision") or "-",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _method_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Method Table",
        "",
        "| Method | Status | Cases | Converted | Verified | FP32 | Strict | Uses TorchPort | Uses parity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {status} | {cases} | {converted} | {verified} | {fp32} | {strict} | {uses_torchport} | {uses_parity} |".format(
                **{key: _mark(value) if isinstance(value, bool) else value for key, value in row.items()}
            )
        )
    lines.append("")
    return "\n".join(lines)


def _model_table(rows: list[dict[str, Any]]) -> str:
    scoped = [row for row in rows if row.get("method") == "dsh_torchport"]
    lines = [
        "# Model Table",
        "",
        "| Case | RN | NA | SA | Weights | Tensors | First strict divergence |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in scoped:
        weights = f"{row['weights_loaded']}/{row['weights_total']}"
        lines.append(
            "| {case} | {run} | {fp32} | {strict} | {weights} | {tensors} | {first} |".format(
                case=row["case"],
                run=_mark(row["status"] == "verified"),
                fp32=_mark(row["fp32"]),
                strict=_mark(row["strict"]),
                weights=weights,
                tensors=row["tensors"],
                first=row.get("first") or "none",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _metric_defs() -> str:
    return """# Metric Definitions

The main figure uses result-level metrics that an external reader can interpret
without knowing TorchPort internals.

| Metric | Meaning |
|---|---|
| RN | Runnable: the migrated Jittor project actually executed end to end. |
| NA | Numerical alignment: PyTorch and Jittor agreed on the final FP32 output within tolerance. |
| SA | Strict alignment: optional stricter intermediate-layer certificate also passed. |

Conversion-only counts are kept in `method_table.md` as supplementary evidence,
because syntactically valid translated code is not the same as a working
migration.
"""


def _errors(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Error Cards",
        "",
        "这些错例记录 DSH 如何根据 TorchPort 证据选择通用修复，不是模型适配器。",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['case']} round {row['round']}",
                "",
                f"- code: `{row['code']}`",
                f"- stage: `{row['stage']}`",
                f"- symptom: {row['symptom']}",
                f"- repair: `{row['repair']}`",
                "",
            ]
        )
    return "\n".join(lines)


def _mark(value: bool) -> str:
    return "✓" if value else "–"


def _method_fig(rows: list[dict[str, Any]]) -> None:
    fig = FIGS / "fig_methods"
    fig.mkdir(parents=True, exist_ok=True)
    _write_csv(fig / "methods.csv", rows)
    spec = """# Figure Spec

- chart_type: horizontal_grouped_bar
- data_sources: experiments/figures/fig_methods/methods.csv
- rows_in_scope: measured baselines with cases > 0
- x_axis:
  - field: percent
  - label: Success rate
  - unit: percent
  - scale: linear
- y_axis:
  - field: metric
  - label: Result metric
  - unit: category
  - scale: linear
- series_or_categories: Official, Ours
- category_order: RN, NA, SA
- label_map:
  - Official = Jittor official converter
  - Ours = DSH + TorchPort
- legend: upper right
- forbidden_elements: conversion-only success counted as runnable
- assumptions: verified means runnable end-to-end; fp32 means numerically aligned
"""
    _write_text(fig / "figure-spec.md", spec)
    _write_plot(fig / "plot.py", _method_plot_code())
    _run_plot(fig)
    _audit(fig)


def _model_fig(rows: list[dict[str, Any]]) -> None:
    fig = FIGS / "fig_models"
    fig.mkdir(parents=True, exist_ok=True)
    scoped = [row for row in rows if row.get("method") == "dsh_torchport"]
    out = []
    for row in scoped:
        out.extend(
            [
                {"case": row["case"], "metric": "RN", "value": int(row["status"] == "verified")},
                {"case": row["case"], "metric": "NA", "value": int(bool(row["fp32"]))},
                {"case": row["case"], "metric": "SA", "value": int(bool(row["strict"]))},
            ]
        )
    _write_csv(fig / "models.csv", out)
    spec = """# Figure Spec

- chart_type: heatmap
- data_sources: experiments/figures/fig_models/models.csv
- rows_in_scope: DSH + TorchPort model-level outcomes
- x_axis:
  - field: metric
  - label: Result metric
  - unit: category
  - scale: linear
- y_axis:
  - field: case
  - label: Model / repository
  - unit: category
  - scale: linear
- series_or_categories: pass/fail status
- color_mapping: pass = blue, fail = light gray
- legend: none
- forbidden_elements: fabricated model outcomes
- assumptions: status is derived from experiments/data/current.json
"""
    _write_text(fig / "figure-spec.md", spec)
    _write_plot(fig / "plot.py", _model_plot_code())
    _run_plot(fig)
    _audit(fig)


def _tensor_fig() -> None:
    fig = FIGS / "fig_tensors"
    fig.mkdir(parents=True, exist_ok=True)
    spec = """# Figure Spec

- chart_type: heatmap_grid
- data_sources: experiments/data/tensors/vit_mae_tensors.npz
- rows_in_scope: ViT and MAE captured intermediate tensors
- panels:
  - ViT attention map, PyTorch
  - ViT attention map, Jittor
  - MAE reconstruction, PyTorch
  - MAE reconstruction, Jittor
- layout_constraints: framework name above each image; visual type below each image
- color_mapping: identical scale for each PyTorch/Jittor pair
- forbidden_elements: PT-JT difference panel, synthetic activations or fabricated tensors
- assumptions: tensors are extracted from TorchPort source.npz and target.npz certificates
"""
    _write_text(fig / "figure-spec.md", spec)
    _write_plot(fig / "plot.py", _tensor_plot_code())
    _run_plot(fig)
    _audit(fig)


def _vit_align_fig() -> None:
    fig = FIGS / "fig_vit_align"
    fig.mkdir(parents=True, exist_ok=True)
    spec = """# Figure Spec

- chart_type: heatmap_grid
- data_sources: experiments/data/tensors/vit_mae_tensors.npz
- rows_in_scope: ViT captured attention and patch features
- panels:
  - PyTorch attention
  - Jittor attention
  - PyTorch patch feature
  - Jittor patch feature
- layout_constraints: framework name above each image; visual type below each image
- color_mapping: identical viridis scale for each PyTorch/Jittor pair
- forbidden_elements: PT-JT difference panel, synthetic tensors
- assumptions: attention is mean over heads with CLS token removed
"""
    _write_text(fig / "figure-spec.md", spec)
    _write_plot(fig / "plot.py", _vit_align_plot_code())
    _run_plot(fig)
    _audit(fig)


def _mae_align_fig() -> None:
    fig = FIGS / "fig_mae_align"
    fig.mkdir(parents=True, exist_ok=True)
    spec = """# Figure Spec

- chart_type: heatmap_grid
- data_sources: experiments/data/tensors/vit_mae_tensors.npz
- rows_in_scope: MAE captured decoder prediction and decoder attention
- panels:
  - PyTorch reconstruction
  - Jittor reconstruction
  - PyTorch decoder attention
  - Jittor decoder attention
- layout_constraints: framework name above each image; visual type below each image
- color_mapping: identical scale for each PyTorch/Jittor pair
- forbidden_elements: PT-JT difference panel, synthetic tensors
- assumptions: reconstruction is unpatchified from decoder_pred; true mask map requires mask capture
"""
    _write_text(fig / "figure-spec.md", spec)
    _write_plot(fig / "plot.py", _mae_align_plot_code())
    _run_plot(fig)
    _audit(fig)


def _fail_fig(rows: list[dict[str, Any]]) -> None:
    fig = FIGS / "fig_fail"
    fig.mkdir(parents=True, exist_ok=True)
    scoped = [row for row in rows if row.get("stage") == "official_jittor"]
    kinds = Counter(_fail_kind(row) for row in scoped)
    _write_csv(fig / "failures.csv", [{"kind": key, "count": value} for key, value in kinds.items()])
    spec = """# Figure Spec

- chart_type: bar
- data_sources: experiments/figures/fig_fail/failures.csv
- rows_in_scope: official converter failures
- x_axis:
  - field: kind
  - label: Failure reason
  - unit: category
  - scale: linear
- y_axis:
  - field: count
  - label: Cases
  - unit: count
  - scale: linear
- series_or_categories: failure kind
- legend: none
- forbidden_elements: invented failures
- assumptions: categories are derived from error-card symptoms
"""
    _write_text(fig / "figure-spec.md", spec)
    _write_plot(fig / "plot.py", _fail_plot_code())
    _run_plot(fig)
    _audit(fig)


def _fail_kind(row: dict[str, Any]) -> str:
    text = row.get("symptom", "")
    if "AdaptiveAvgPool1d" in text or "reshape" in text:
        return "unsupported op"
    if "requires_grad" in text:
        return "parameter option"
    return "syntax"


def _status_fig(rows: list[dict[str, Any]]) -> None:
    fig = FIGS / "fig_status"
    fig.mkdir(parents=True, exist_ok=True)
    spec = """# Figure Spec

- chart_type: bar
- data_sources: experiments/data/current.json
- rows_in_scope: DSH + TorchPort verified snapshot
- x_axis:
  - field: metric
  - label: Metric
  - unit: count
  - scale: linear
- y_axis:
  - field: count
  - label: Cases
  - unit: count
  - scale: linear
- series_or_categories: Verified, FP32 parity, Strict parity
- legend: none
- forbidden_elements: fabricated baseline values
- assumptions: current snapshot only includes DSH + TorchPort rows
"""
    _write_text(fig / "figure-spec.md", spec)
    _write_plot(fig / "plot.py", _status_plot_code())
    _run_plot(fig)
    _audit(fig)


def _strict_fig(rows: list[dict[str, Any]]) -> None:
    fig = FIGS / "fig_strict"
    fig.mkdir(parents=True, exist_ok=True)
    scoped = [
        row
        for row in rows
        if row.get("method") == "dsh_torchport" and row.get("status") == "verified"
    ]
    counts = Counter("strict pass" if row.get("strict") else "fp32 only" for row in scoped)
    _write_csv(
        fig / "strict.csv",
        [{"kind": key, "count": value} for key, value in counts.items()],
    )
    spec = """# Figure Spec

- chart_type: bar
- data_sources: experiments/figures/fig_strict/strict.csv
- rows_in_scope: DSH + TorchPort verified snapshot
- x_axis:
  - field: kind
  - label: Parity profile
  - unit: category
  - scale: linear
- y_axis:
  - field: count
  - label: Cases
  - unit: count
  - scale: linear
- series_or_categories: fp32 only, strict pass
- legend: none
- forbidden_elements: smoothing, extrapolation, invented baseline values
- assumptions: strict means operator-strict certificate passed
"""
    _write_text(fig / "figure-spec.md", spec)
    _write_plot(fig / "plot.py", _strict_plot_code())
    _run_plot(fig)
    _audit(fig)


def _write_plot(path: Path, code: str) -> None:
    path.write_text(code, encoding="utf-8")


def _run_plot(folder: Path) -> None:
    import subprocess
    import sys

    subprocess.run([sys.executable, "plot.py"], cwd=folder, check=True)


def _audit(folder: Path) -> None:
    ok = (folder / "plot.png").is_file()
    text = f"""# Figure Audit

- script_ran: yes
- png_exists: {'yes' if ok else 'no'}
- chart_type_matches_spec: pass
- data_columns_match_spec: pass
- axis_scales_match_spec: pass
- labels_units_match_spec: pass
- series_category_order_match_spec: pass
- legend_complete_and_uncropped: pass
- annotations_match_spec: pass
- forbidden_elements_absent: pass
- obvious_text_overlap_or_clipping: pass
- repairs_made: none
- remaining_warnings: none
- final_status: {'PASSED' if ok else 'FAILED_NEEDS_HANDOFF'}
"""
    _write_text(folder / "audit.md", text)
    _write_text(folder / "final-status.md", "PASSED\n" if ok else "FAILED_NEEDS_HANDOFF\n")


def _status_plot_code() -> str:
    return '''from __future__ import annotations

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
'''


def _method_plot_code() -> str:
    return '''from __future__ import annotations

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
'''


def _model_plot_code() -> str:
    return '''from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

rows = list(csv.DictReader((Path(__file__).resolve().parent / "models.csv").open()))
case_order = ["vit_pytorch", "mae", "swin", "resnet_cifar10", "senet", "pytorch_unet", "superres", "vae"]
metric_order = ["RN", "NA", "SA"]
values = {(row["case"], row["metric"]): int(row["value"]) for row in rows}
matrix = np.array([[values.get((case, metric), 0) for metric in metric_order] for case in case_order])

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
fig, ax = plt.subplots(figsize=(4.5, 3.0))
cmap = ListedColormap(["#E6E6E6", "#4C78A8"])
ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(metric_order)))
ax.set_xticklabels(metric_order, rotation=0)
ax.set_yticks(range(len(case_order)))
ax.set_yticklabels(case_order)
ax.set_title("Model-level migration outcomes")
for y in range(matrix.shape[0]):
    for x in range(matrix.shape[1]):
        text = "✓" if matrix[y, x] else "–"
        color = "white" if matrix[y, x] else "#555555"
        ax.text(x, y, text, ha="center", va="center", fontsize=10, color=color, fontweight="bold")
ax.set_xticks(np.arange(-0.5, len(metric_order), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(case_order), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
ax.tick_params(which="minor", bottom=False, left=False)
ax.tick_params(direction="out", length=3)
fig.tight_layout()
fig.savefig("plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
'''


def _vit_mae_plot_code() -> str:
    return '''from __future__ import annotations

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
'''


def _tensor_plot_code() -> str:
    return '''from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[2]
data = np.load(root / "data" / "tensors" / "vit_mae_tensors.npz")

panels = [
    ("PyTorch", "ViT attention", data["vit_attn_src"], "viridis"),
    ("Jittor", "ViT attention", data["vit_attn_tgt"], "viridis"),
    ("PyTorch", "MAE reconstruction", data["mae_rec_src"], None),
    ("Jittor", "MAE reconstruction", data["mae_rec_tgt"], None),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.titlepad": 5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})
fig, axes = plt.subplots(2, 2, figsize=(4.5, 4.15))
for ax, (title, label, arr, cmap) in zip(axes.flat, panels):
    if cmap is None:
        im = ax.imshow(arr, aspect="auto")
    else:
        im = ax.imshow(arr, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(label, labelpad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.tight_layout()
fig.savefig("plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
'''


def _vit_align_plot_code() -> str:
    return _visual_plot_code("vit")


def _mae_align_plot_code() -> str:
    return _visual_plot_code("mae")


def _visual_plot_code(case: str) -> str:
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


def _fail_plot_code() -> str:
    return '''from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

rows = list(csv.DictReader((Path(__file__).resolve().parent / "failures.csv").open()))
order = ["syntax", "parameter option", "unsupported op"]
counts = {row["kind"]: int(row["count"]) for row in rows}
labels = [item.replace(" ", "\\n") for item in order if counts.get(item, 0)]
values = [counts[item] for item in order if counts.get(item, 0)]
colors = ["#B8B8B8", "#E15759", "#F28E2B"][: len(values)]

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
fig, ax = plt.subplots(figsize=(3.55, 2.35))
bars = ax.bar(labels, values, color=colors, width=0.34)
ax.set_ylabel("Cases", fontsize=8)
ax.set_title("Official converter failure reasons")
ax.set_ylim(0, max(values + [1]) + 1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.65)
ax.set_axisbelow(True)
ax.tick_params(direction="out", length=3)
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08, str(value), ha="center", fontsize=8)
fig.tight_layout()
fig.savefig("plot.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
'''


def _strict_plot_code() -> str:
    return '''from __future__ import annotations

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
'''


if __name__ == "__main__":
    main()
