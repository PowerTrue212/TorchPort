from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAFT = ROOT / "RAFT"
DEFAULT_RUN_ID = "raft-v1-2026-08-25-v100"
METHODS = ("codex", "dsh", "dsh_torchport")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_path(value: str, raft_root: Path) -> str:
    normalized = value.replace("\\", "/")
    roots = (
        str(raft_root.resolve()).replace("\\", "/") + "/",
        "/root/data-fs/RAFT/",
        "/data-fs/RAFT/",
    )
    for prefix in roots:
        if normalized.lower().startswith(prefix.lower()):
            return "data-fs://RAFT/" + normalized[len(prefix) :]
    if re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("/root/"):
        return "redacted-local-path://" + Path(normalized).name
    return value


def sanitize(value: Any, raft_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item, raft_root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, raft_root) for item in value]
    if isinstance(value, str):
        return logical_path(value, raft_root)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_json(source: Path, target: Path, raft_root: Path) -> None:
    write_json(target, sanitize(json.loads(source.read_text(encoding="utf-8")), raft_root))


def copy_jsonl(source: Path, target: Path, raft_root: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.dumps(sanitize(json.loads(line), raft_root), ensure_ascii=False))
    target.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")


def export_compact_evidence(raft_root: Path, bundle: Path) -> None:
    copy_json(raft_root / "experiment.json", bundle / "protocol.json", raft_root)
    copy_json(raft_root / "outputs" / "summary.json", bundle / "summary.json", raft_root)
    for method in METHODS:
        source = raft_root / "results" / method
        target = bundle / "methods" / method
        for name in ("result.json", "termination.json", "agent_metrics.json"):
            if (source / name).is_file():
                copy_json(source / name, target / name, raft_root)
        if (source / "rounds.jsonl").is_file():
            copy_jsonl(source / "rounds.jsonl", target / "rounds.jsonl", raft_root)
        if (source / "isolation" / "latest.json").is_file():
            copy_json(source / "isolation" / "latest.json", target / "isolation.json", raft_root)

    trace_root = raft_root / "results" / "dsh_torchport" / "traces"
    torchport_out = bundle / "torchport"
    for name in (
        "certificate.json",
        "certification_report.json",
        "compare_fp32.json",
        "compare_strict.json",
        "plan.json",
        "scan.json",
    ):
        if (trace_root / name).is_file():
            copy_json(trace_root / name, torchport_out / name, raft_root)
    if (trace_root / "strict_rounds.jsonl").is_file():
        copy_jsonl(trace_root / "strict_rounds.jsonl", torchport_out / "strict_rounds.jsonl", raft_root)

    report = raft_root / "outputs" / "deepseek_strict_analysis.md"
    if report.is_file():
        target = bundle / "reports" / report.name
        target.parent.mkdir(parents=True, exist_ok=True)
        text = report.read_text(encoding="utf-8")
        text = text.replace(str(raft_root), "data-fs://RAFT").replace("/root/data-fs/RAFT", "data-fs://RAFT")
        target.write_text(text, encoding="utf-8")


def write_figure_data(summary: dict[str, Any], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_fields = [
        "method",
        "display_name",
        "status",
        "success",
        "rn",
        "na",
        "max_error",
        "mean_error",
        "mean_framework_flow_difference",
        "p99_framework_flow_difference",
        "max_framework_flow_difference",
        "rounds",
        "evaluator_seconds",
        "agent_elapsed_seconds",
        "agent_elapsed_kind",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "non_cache_tokens",
    ]
    rows = []
    for result in summary["results"]:
        metrics = result.get("agent_metrics", {})
        tokens = metrics.get("token_usage_raw", {})
        known = [tokens.get(key) for key in ("input_tokens", "output_tokens", "reasoning_tokens")]
        rows.append(
            {
                "method": result["method"],
                "display_name": result["display_name"],
                "status": result["status"],
                "success": result["success"],
                "rn": result["rn"],
                "na": result["na"],
                "max_error": result.get("max_error"),
                "mean_error": result.get("mean_error"),
                "mean_framework_flow_difference": result.get("mean_framework_flow_difference"),
                "p99_framework_flow_difference": result.get("p99_framework_flow_difference"),
                "max_framework_flow_difference": result.get("max_framework_flow_difference"),
                "rounds": result["rounds"],
                "evaluator_seconds": result.get("elapsed_seconds"),
                "agent_elapsed_seconds": metrics.get("agent_elapsed_seconds", metrics.get("wall_elapsed_seconds")),
                "agent_elapsed_kind": metrics.get("agent_elapsed_kind"),
                "input_tokens": tokens.get("input_tokens"),
                "output_tokens": tokens.get("output_tokens"),
                "reasoning_tokens": tokens.get("reasoning_tokens"),
                "cache_read_tokens": tokens.get("cache_read_tokens"),
                "non_cache_tokens": sum(known) if all(value is not None for value in known) else None,
            }
        )
    with (figure_dir / "data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(rows)


def raw_manifest(raft_root: Path) -> list[dict[str, Any]]:
    candidates = [
        "shared/checkpoints/raft-things.pth",
        "shared/prepared/image1.npy",
        "shared/prepared/image2.npy",
        "shared/prepared/flow_low_ref.npy",
        "shared/prepared/flow_up_ref.npy",
        "shared/prepared/weights_fp32.npz",
        *[f"results/{method}/flow_up.npy" for method in METHODS],
        "results/dsh_torchport/traces/source.npz",
        "results/dsh_torchport/traces/target.npz",
    ]
    records = []
    for relative in candidates:
        path = raft_root / relative
        record: dict[str, Any] = {
            "logical_path": f"data-fs://RAFT/{relative}",
            "retention": "V100/data-fs only; do not commit",
            "present_in_export_source": path.is_file(),
        }
        if path.is_file():
            record.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
        records.append(record)
    return records


def finalize_manifest(bundle: Path, raft_root: Path, run_id: str) -> None:
    public_files = []
    for path in sorted(item for item in bundle.rglob("*") if item.is_file() and item.name != "manifest.json"):
        public_files.append(
            {
                "path": path.relative_to(bundle).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_json(
        bundle / "manifest.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "storage_policy": {
                "git": "compact sanitized evidence and selected figures",
                "data_fs": "weights, fixtures, references, workspaces, arrays, full traces, and session logs",
            },
            "public_files": public_files,
            "raw_artifacts": raw_manifest(raft_root),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raft-root", type=Path, default=DEFAULT_RAFT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    raft_root = args.raft_root.resolve()
    bundle = ROOT / "experiments" / "results" / args.run_id
    figure_dir = ROOT / "experiments" / "figures" / "fig_raft"
    bundle.mkdir(parents=True, exist_ok=True)
    export_compact_evidence(raft_root, bundle)
    summary = json.loads((raft_root / "outputs" / "summary.json").read_text(encoding="utf-8"))
    write_figure_data(summary, figure_dir)
    for name in ("flows.png", "errors.png", "method_results.png"):
        source = raft_root / "outputs" / name
        if source.is_file():
            shutil.copy2(source, figure_dir / name)
    write_json(
        figure_dir / "audit.json",
        {
            "source": f"experiments/results/{args.run_id}/summary.json",
            "generated_from": "frozen PyTorch reference and common-evaluator Jittor outputs",
            "shared_flow_color_wheel": True,
            "shared_error_color_scale": True,
            "feature_source": "experiments/data/tensors/raft_features.npz",
            "feature_reduction": "sqrt(mean(channel_value ** 2)) for each spatial location",
            "shared_paired_feature_scale": True,
            "feature_data_is_synthetic": False,
            "agent_cost_plot_included": False,
            "missing_agent_costs_imputed": False,
            "pure_dsh_terminal_status": "stopped_failed",
        },
    )
    write_json(
        bundle / "audit.json",
        {
            "run_id": args.run_id,
            "success_definition": "RN == true AND NA == true",
            "method_status": {item["method"]: item["status"] for item in summary["results"]},
            "caveats": [
                "pure DSH was user-stopped and is reported as stopped_failed with RN=true, NA=false.",
                "The recorded pure DSH session exposed TorchPort tool schemas but made zero forbidden TorchPort calls; it is not a clean isolated baseline.",
                "DSH + TorchPort passes the common FP32 evaluator but TorchPort strict remains false.",
                "Complete agent elapsed/token accounting is currently available only for pure DSH; missing values are not zero.",
            ],
        },
    )
    (bundle / "README.md").write_text(
        "# RAFT v1 V100 experiment evidence\n\n"
        "This directory is the compact, sanitized Git-facing evidence bundle. "
        "Large checkpoints, fixtures, references, workspaces, flow arrays, NPZ captures, "
        "and full session logs remain in V100/data-fs and are referenced by `manifest.json`.\n\n"
        "- Codex: passed (RN=true, NA=true).\n"
        "- pure DSH: stopped_failed (RN=true, NA=false); the final edited workspace was not evaluated.\n"
        "- DSH + TorchPort: common FP32 evaluator passed; TorchPort strict remains false.\n\n"
        "Missing agent time/token fields mean not recorded, never zero. See `audit.json` for the "
        "method-integrity and reporting caveats.\n",
        encoding="utf-8",
    )
    finalize_manifest(bundle, raft_root, args.run_id)
    print(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
