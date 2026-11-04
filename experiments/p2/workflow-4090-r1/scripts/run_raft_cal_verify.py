"""Bind the frozen r6 RAFT train oracle bundle to a target repo and run one hidden verify.

The oracle identity does not depend on target.repo, so binding only changes the target side.
Writes a new task JSON (mode 'x', never overwrites) and a new verify report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/RAFT-Benchmark-v1/oracle-bundle-train-three-r1-r6/src")
from torchport.contracts import load_task  # noqa: E402
from torchport.verification import verify  # noqa: E402

ROOT = Path("/root/autodl-tmp/RAFT-Benchmark-v1")
BUNDLE = ROOT / "oracle-bundle-train-three-r1-r6"
EVIDENCE = BUNDLE / "evidence"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial")
    parser.add_argument("repo", nargs="?", default=None)
    parser.add_argument("--task-file", default=None)
    parser.add_argument("--pilot-baseline", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if not args.trial.replace("-", "").isalnum():
        raise ValueError("trial")
    if args.task_file:
        task_path = Path(args.task_file)
    else:
        if not args.repo:
            raise SystemExit("repo or --task-file required")
        raw = json.loads((BUNDLE / "task.json").read_text())
        raw["target"]["repo"] = str(Path(args.repo).resolve())
        if args.pilot_baseline:
            import hashlib

            baseline = Path(args.pilot_baseline).resolve()
            raw["protocol"].update(
                train_envelope="train-repeat-envelope-v1-pilot",
                train_noise_baseline=str(baseline),
                train_noise_baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
            )
        task_path = BUNDLE / f"task-{args.trial}.json"
        with task_path.open("x") as stream:
            json.dump(raw, stream, indent=2)
    task = load_task(task_path)
    report = verify(task, EVIDENCE, trial=args.trial)
    out = Path(args.out) if args.out else ROOT / "evidence" / f"verify-train-{args.trial}.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(report, indent=2) + "\n")
    result = (report.get("results") or [{}])[0]
    print(json.dumps({
        "trial": args.trial,
        "task_digest": task["digest"],
        "task_file": str(task_path),
        "passed": report.get("passed"),
        "failure_code": report.get("failure_code"),
        "summary": report.get("summary"),
        "repeats": report.get("repeats"),
        "mismatch": report.get("mismatch"),
        "envelope": result.get("envelope_metrics"),
        "weights": (result.get("evidence") or {}).get("weights"),
        "physical": report.get("physical_cost"),
        "report": str(out),
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
