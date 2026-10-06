"""通过同一 normal capture 路径测量通过验收的 candidate 性能。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from torchport.execution import add_physical_cost, run_physical_cost, write_json
from torchport.oracle import build_oracle, evaluate, run_side
from torchport.task import load_task


def measure_performance(
    task: dict[str, Any],
    verification: dict[str, Any],
    root: str | Path,
    *,
    trial: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if not verification.get("passed") or verification.get("task_digest") != task["digest"]:
        raise ValueError("performance 只接受同一 TaskSpec 下通过的 official verify")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup 必须非负且 iterations 必须为正数")

    root = Path(root).resolve()
    oracle = build_oracle(task, root)
    if not oracle.get("ok"):
        raise ValueError(f"performance oracle 不可用：{oracle.get('reason')}")
    request = {"warmup": warmup, "iterations": iterations}
    runs = [
        run_side(
            task,
            "target",
            root,
            trial=trial,
            mode="normal",
            oracle=Path(oracle["dir"]),
            performance=request,
        )
        for _ in range(2)
    ]
    results = [
        evaluate(oracle, run, task["protocol"], task["target_identity"])
        for run in runs
    ]
    if not all(result.get("passed") for result in results):
        raise ValueError("performance 前后的正常输出未通过数值验收")

    cold, warm = runs
    warm_metrics = warm["capture"]["performance"]
    measurement_cost = add_physical_cost(
        *(run_physical_cost(run, task["target"]["device"]) for run in runs)
    )
    return {
        "schema_version": 2,
        "kind": "torchport.performance",
        "task_id": task["task_id"],
        "task_digest": task["digest"],
        "candidate_identity": task["target_identity"],
        "trial_id": trial,
        "mode": "normal",
        "trace": False,
        "passed_before_measurement": True,
        "correctness_preserved": True,
        "warmup": warmup,
        "iterations": iterations,
        "batch_size": warm_metrics["batch_size"],
        "cold_start_seconds": _first_output_seconds(cold),
        "warm_start_seconds": _first_output_seconds(warm),
        "steady_latency_ms": warm_metrics["steady_latency_ms"],
        "throughput_per_second": warm_metrics["throughput_per_second"],
        "memory": "unsupported",
        "latency_samples_ms": warm_metrics["latency_ms"],
        "physical_evaluation_cost": {
            "setup": oracle.get("physical_cost"),
            "measurement": measurement_cost,
            "total": add_physical_cost(oracle.get("physical_cost"), measurement_cost),
        },
        "runs": [run["run_id"] for run in runs],
    }


def _first_output_seconds(run: dict[str, Any]) -> float:
    started = float(run["execution"]["started"])
    first = float(run["capture"]["performance"]["first_output_at"])
    return max(0.0, first - started)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure a verified TorchPort candidate")
    parser.add_argument("task", type=Path)
    parser.add_argument("verification", type=Path)
    parser.add_argument("--root", type=Path, default=Path(".torchport/performance"))
    parser.add_argument("--trial", default="performance")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        task = load_task(args.task)
        verification = json.loads(args.verification.read_text("utf-8"))
        report = measure_performance(
            task,
            verification,
            args.root,
            trial=args.trial,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"performance: {exc}") from exc
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
