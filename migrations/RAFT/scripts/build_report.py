from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, load_config, verify_manifest, write_json

sys.path.insert(0, str(ROOT / "source" / "princeton-vl-raft" / "core"))
from utils.flow_viz import flow_uv_to_colors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen RAFT result tables and visuals")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def load_results(config: dict[str, Any], allow_incomplete: bool) -> list[dict[str, Any]]:
    values = []
    for method in config["methods"]:
        method_dir = ROOT / "results" / method["id"]
        path = method_dir / "result.json"
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
        elif allow_incomplete:
            result = {
                "method": method["id"],
                "rn": False,
                "na": False,
                "max_error": None,
                "mean_error": None,
                "rounds": 0,
                "failure": {"type": "MissingResult", "message": "not evaluated"},
            }
        else:
            raise FileNotFoundError(f"missing final result for {method['display_name']}: {path}")
        termination_path = method_dir / "termination.json"
        metrics_path = method_dir / "agent_metrics.json"
        if termination_path.is_file():
            termination = json.loads(termination_path.read_text(encoding="utf-8"))
            result["termination"] = termination
            result["status"] = termination["status"]
            result["success"] = False
        else:
            result["status"] = "passed" if result["rn"] and result["na"] else "failed"
            result["success"] = bool(result["rn"] and result["na"])
        if metrics_path.is_file():
            result["agent_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        result["display_name"] = method["display_name"]
        values.append(result)
    return values


def load_method_flows(results: list[dict[str, Any]]) -> list[np.ndarray | None]:
    flows: list[np.ndarray | None] = []
    for result in results:
        path = ROOT / "results" / result["method"] / "flow_up.npy"
        flows.append(np.load(path, allow_pickle=False) if result["rn"] and path.is_file() else None)
    return flows


def flow_rgb(flow: np.ndarray, maximum: float) -> np.ndarray:
    value = flow[0].astype(np.float64, copy=False)
    scale = max(maximum, np.finfo(np.float64).eps)
    return flow_uv_to_colors(value[0] / scale, value[1] / scale, convert_to_bgr=False)


def write_tables(results: list[dict[str, Any]], output: Path) -> None:
    fields = ["Method", "RN", "NA", "Max Error", "Mean Error", "Rounds"]
    rows = []
    for result in results:
        rows.append(
            {
                "Method": result["display_name"],
                "RN": str(bool(result["rn"])),
                "NA": str(bool(result["na"])),
                "Max Error": "" if result.get("max_error") is None else f"{result['max_error']:.9g}",
                "Mean Error": ""
                if result.get("mean_error") is None
                else f"{result['mean_error']:.9g}",
                "Rounds": int(result["rounds"]),
            }
        )
    with (output / "method_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Method | RN | NA | Max Error | Mean Error | Rounds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Method']} | {row['RN']} | {row['NA']} | "
            f"{row['Max Error'] or '—'} | {row['Mean Error'] or '—'} | {row['Rounds']} |"
        )
    terminal = [result for result in results if result.get("status") == "stopped_failed"]
    for result in terminal:
        lines.extend(
            [
                "",
                f"- {result['display_name']} terminal status: `stopped_failed` "
                f"(`success=false`; last evaluator RN={result['rn']}, NA={result['na']}).",
            ]
        )
    (output / "method_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_agent_costs(results: list[dict[str, Any]], output: Path) -> None:
    fields = [
        "Method",
        "Agent Turns",
        "Model Calls",
        "Tool Calls",
        "Agent Elapsed Seconds",
        "Elapsed Kind",
        "Evaluator Seconds",
        "Input Tokens",
        "Output Tokens",
        "Cache Read Tokens",
        "Reasoning Tokens",
    ]
    rows = []
    for result in results:
        metrics = result.get("agent_metrics")
        tokens = metrics.get("token_usage_raw", {}) if metrics else {}
        rows.append(
            {
                "Method": result["display_name"],
                "Agent Turns": metrics.get("agent_turns") if metrics else None,
                "Model Calls": metrics.get("model_calls") if metrics else None,
                "Tool Calls": metrics.get("tool_calls") if metrics else None,
                "Agent Elapsed Seconds": (
                    metrics.get("agent_elapsed_seconds", metrics.get("wall_elapsed_seconds"))
                    if metrics
                    else None
                ),
                "Elapsed Kind": metrics.get("agent_elapsed_kind") if metrics else None,
                "Evaluator Seconds": result.get("elapsed_seconds"),
                "Input Tokens": tokens.get("input_tokens"),
                "Output Tokens": tokens.get("output_tokens"),
                "Cache Read Tokens": tokens.get("cache_read_tokens"),
                "Reasoning Tokens": tokens.get("reasoning_tokens"),
            }
        )
    with (output / "agent_costs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Method | Agent Turns | Model Calls | Tool Calls | Agent Elapsed Seconds | Elapsed Kind | Evaluator Seconds | Input Tokens | Output Tokens | Cache Read Tokens | Reasoning Tokens |",
        "|---|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                "not_recorded" if row[field] is None else str(row[field]) for field in fields
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Agent elapsed is the primary cost measure. `session_wall` may include idle or human-paused time; evaluator time is listed separately. Token counters are optional provider-native data, and missing values mean not recorded, not zero.",
        ]
    )
    (output / "agent_costs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_flows(
    results: list[dict[str, Any]], reference: np.ndarray, flows: list[np.ndarray | None], output: Path
) -> None:
    values = [reference, *[flow for flow in flows if flow is not None]]
    maximum = max(float(np.sqrt(np.sum(np.square(flow.astype(np.float64)), axis=1)).max()) for flow in values)
    panels: list[np.ndarray | None] = [reference, *flows]
    titles = ["PyTorch", *[result["display_name"] for result in results]]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    for axis, title, flow in zip(axes, titles, panels):
        axis.set_title(title, fontsize=12, weight="bold")
        axis.axis("off")
        if flow is None:
            axis.set_facecolor("#eeeeee")
            axis.text(0.5, 0.5, "forward failed", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.imshow(flow_rgb(flow, maximum))
    figure.suptitle(f"Optical flow — common color wheel, max magnitude {maximum:.4g}")
    figure.savefig(output / "flows.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_errors(
    results: list[dict[str, Any]], reference: np.ndarray, flows: list[np.ndarray | None], output: Path
) -> None:
    errors = [
        None
        if flow is None
        else np.sqrt(np.sum(np.square(flow.astype(np.float64) - reference.astype(np.float64)), axis=1))[0]
        for flow in flows
    ]
    available = [error for error in errors if error is not None]
    maximum = max((float(error.max()) for error in available), default=0.0)
    color_max = max(maximum, 1e-12)
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), constrained_layout=True)
    image = None
    for axis, result, error in zip(axes, results, errors):
        axis.set_title(result["display_name"], fontsize=12, weight="bold")
        axis.axis("off")
        if error is None:
            axis.set_facecolor("#eeeeee")
            axis.text(0.5, 0.5, "forward failed", ha="center", va="center", transform=axis.transAxes)
        else:
            image = axis.imshow(error, cmap="magma", vmin=0.0, vmax=color_max)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.82, label="Framework Flow Difference")
    figure.savefig(output / "errors.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_method_results(results: list[dict[str, Any]], output: Path) -> None:
    colors = {
        "Codex": "#4C78A8",
        "DSH": "#E45756",
        "DSH + TorchPort": "#72A35A",
    }
    figure, axis = plt.subplots(figsize=(8.8, 4.2))
    label_offsets = {"Codex": (9, 8), "DSH + TorchPort": (9, 8)}
    for result in results:
        name = result["display_name"]
        if name == "DSH":
            continue
        rounds = int(result["rounds"])
        passed = bool(result["success"])
        color = colors[name]
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
        marker_size = 6 if name == "DSH + TorchPort" else 8.5
        axis.plot(x_values, stages, color=color, marker=marker, markersize=marker_size, linewidth=2.2)
        if not passed:
            axis.scatter(x_values[-1], stages[-1], s=125, facecolor="white", edgecolor=color, linewidth=2.4, marker="X", zorder=4)
        label = name if passed else f"{name} · FAIL"
        axis.annotate(label, (x_values[-1], stages[-1]), xytext=label_offsets[name], textcoords="offset points", fontsize=10)
    axis.set_xticks(np.arange(0, 6))
    axis.set_xlim(-0.15, 5.75)
    axis.set_yticks([0, 1], ["RN", "NA"])
    axis.set_ylim(-0.24, 1.24)
    axis.set_xlabel("Repair round")
    axis.set_ylabel("Highest stage reached")
    axis.set_title("RAFT repair trajectories")
    axis.grid(True, linestyle=":", linewidth=0.65, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3)
    figure.tight_layout()
    figure.savefig(output / "method_results.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    config = load_config()
    verify_manifest("reference")
    results = load_results(config, args.allow_incomplete)
    flows = load_method_flows(results)
    reference = np.load(ROOT / "shared" / "prepared" / "flow_up_ref.npy", allow_pickle=False)
    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)

    write_tables(results, output)
    write_agent_costs(results, output)
    write_json(
        output / "summary.json",
        {
            "protocol_version": config["protocol_version"],
            "alignment": config["alignment"],
            "results": results,
        },
    )
    plot_flows(results, reference, flows, output)
    plot_errors(results, reference, flows, output)
    plot_method_results(results, output)
    print(f"wrote flows.png, errors.png, and method_results.png to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
