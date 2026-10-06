"""Validate materialized CLIP/RAFT tasks, CUDA probe, and public/hidden verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from experiments.harness.gateway import HarnessGateway
from experiments.harness.protocol import FeedbackLevel, TrialProtocol
from torchport.execution import write_json
from torchport.oracle import verify
from torchport.task import load_task

PROBE = """\
import json
import os
import jittor as jt

jt.flags.use_cuda = 1
value = (jt.ones((16, 16)) * 3).sum()
jt.sync_all()
print(json.dumps({
    "cuda": int(jt.flags.use_cuda),
    "jittor_home": os.environ.get("JITTOR_HOME"),
    "value": float(value.item()),
}))
"""

INFRASTRUCTURE_FAILURES = {"SOURCE_RUN_FAILED", "TARGET_RUN_FAILED", "ORACLE_CORRUPT"}


def _probe(task_path: Path, output: Path, task_name: str) -> dict[str, Any]:
    gateway = HarnessGateway(
        task_path,
        output / "evidence" / task_name,
        output / "gateway" / task_name,
        TrialProtocol(FeedbackLevel.LOCALIZED, 2, 1800),
        trial_id=f"{task_name}-gpu-probe",
    )
    existing = gateway.call("read_file", {"path": "_torchport_probe.py"})
    old = existing["data"]["text"] if existing["ok"] else None
    created = gateway.call("apply_patch", {"path": "_torchport_probe.py", "old": old, "new": PROBE})
    if not created["ok"]:
        raise RuntimeError(created)
    response = gateway.call("probe", {"path": "_torchport_probe.py"})
    if not response["ok"] or not response["data"]["ok"]:
        raise RuntimeError(response)
    stdout = json.loads(response["data"]["stdout"].strip().splitlines()[-1])
    cost = response["budget"]["physical_evaluation_cost"]
    if stdout["cuda"] != 1 or cost["executions"] != 1 or cost["cuda_execution_seconds"] <= 0:
        raise RuntimeError({"stdout": stdout, "cost": cost})
    return {"response": response["data"], "runtime": stdout, "budget": response["budget"]}


def validate(materialized: Path, output: Path, tasks: tuple[str, ...]) -> dict[str, Any]:
    manifest = json.loads((materialized / "experiment.json").read_text("utf-8"))
    selected = [item for item in manifest["trials"] if item["task"] in tasks]
    target_repos = [
        load_task(materialized / item["public_task"])["target"]["repo"] for item in selected
    ]
    if len(target_repos) != len(set(target_repos)):
        raise RuntimeError("trial candidates are not isolated")

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "torchport.p2-clip-raft-gpu-validation",
        "experiment_id": manifest["experiment_id"],
        "isolation": manifest["isolation"],
        "tasks": {},
    }
    for task_name in tasks:
        item = next(
            value
            for value in selected
            if value["task"] == task_name and value["condition"] == "localized"
        )
        public_path = materialized / item["public_task"]
        hidden_path = materialized / item["hidden_task"]
        public = load_task(public_path)
        hidden = load_task(hidden_path)
        if public["digest"] == hidden["digest"]:
            raise RuntimeError(f"{task_name} public/hidden task identity is equal")
        task_output = output / task_name
        probe = _probe(public_path, task_output, task_name)
        public_report = verify(public, task_output / "verify", trial="gpu-validation")
        hidden_report = verify(hidden, task_output / "verify", trial="gpu-validation")
        value = {
            "public_task_digest": public["digest"],
            "hidden_task_digest": hidden["digest"],
            "candidate_identity": load_task(public_path)["target_identity"],
            "probe": probe,
            "public_verify": public_report,
            "hidden_verify": hidden_report,
        }
        write_json(task_output / "validation.json", value)
        report["tasks"][task_name] = {
            "probe_ok": probe["response"]["ok"],
            "probe_cuda_seconds": probe["budget"]["physical_evaluation_cost"][
                "cuda_execution_seconds"
            ],
            "public_passed": bool(public_report["passed"]),
            "hidden_passed": bool(hidden_report["passed"]),
            "public_evaluated": public_report.get("failure_code") not in INFRASTRUCTURE_FAILURES,
            "hidden_evaluated": hidden_report.get("failure_code") not in INFRASTRUCTURE_FAILURES,
            "public_failure_code": public_report.get("failure_code"),
            "hidden_failure_code": hidden_report.get("failure_code"),
            "public_physical_cost": public_report.get("physical_cost"),
            "hidden_physical_cost": hidden_report.get("physical_cost"),
            "artifact": str(task_output / "validation.json"),
        }
    report["infrastructure_passed"] = all(
        value["probe_ok"] and value["public_evaluated"] and value["hidden_evaluated"]
        for value in report["tasks"].values()
    )
    report["candidate_passed"] = all(
        value["public_passed"] and value["hidden_passed"] for value in report["tasks"].values()
    )
    report["passed"] = report["infrastructure_passed"]
    write_json(output / "validation-summary.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materialized", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task", action="append", choices=("clip", "raft"))
    args = parser.parse_args(argv)
    tasks = tuple(args.task or ("clip", "raft"))
    report = validate(args.materialized.resolve(), args.output.resolve(), tasks)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
