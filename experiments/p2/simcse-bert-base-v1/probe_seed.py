"""Run a controller-only smoke verify before any SimCSE Agent trial."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.harness.remote_executor import RemoteExecutor  # noqa: E402

BENCH = Path("D:/科研/SimCSE-Benchmark-v1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("none", "full"), required=True)
    parser.add_argument("--task", choices=("smoke", "stsb"), default="smoke")
    parser.add_argument("--tool", choices=("run", "verify"), default="verify")
    parser.add_argument("--tasks-dir", type=Path,
                        default=BENCH / "controller/tasks-cache-root-nomkl")
    parser.add_argument("--trial-id", required=True)
    args = parser.parse_args()
    task_path = args.tasks_dir / f"task-{args.task}-{args.condition}.json"
    candidate = BENCH / "workspaces" / args.condition
    control = BENCH / "controller/prep" / args.trial_id
    if control.exists():
        raise FileExistsError(control)
    control.mkdir(parents=True)
    os.environ["TP_EXTRA_ROOTS"] = str(BENCH.resolve())
    executor = RemoteExecutor(
        ROOT, control / "remote-raw", candidate,
        remote={
            "root": "/root/SimCSE-Benchmark-v1/harness",
            "source_python": "/root/SimCSE-Benchmark-v1/source_python.sh",
            "target_python": "/root/data-fs/APT/.venv-jittor/bin/python",
            "python_config": "",
            "system_libstdcxx": "",
            "nvcc_path": "/usr/local/cuda/bin/nvcc",
            "home": "/root/SimCSE-Benchmark-v1/home",
            "target_path": "/usr/local/cuda/bin:/root/data-fs/APT/.venv-jittor/bin:"
                           "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        },
    )
    result = executor.call(args.tool, task_path, trial=args.trial_id)
    (control / f"{args.tool}-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result.get("passed"),
                      "failure": result.get("failure"),
                      "control": str(control)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
