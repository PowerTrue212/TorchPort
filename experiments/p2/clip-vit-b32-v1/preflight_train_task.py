"""Build the CLIP training source oracle through the real remote verify path.

The untouched inference seed deliberately lacks the training entry. A target
RUN_FAILED is expected here; a complete source oracle is the objective.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.harness.remote_executor import RemoteExecutor  # noqa: E402
from torchport.runtime import write_json  # noqa: E402

BENCH = Path("D:/科研/CLIP-Benchmark-v1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=BENCH / "workspaces/full")
    parser.add_argument("--trial", default="clip-train-source-r0")
    args = parser.parse_args()
    controller = args.controller.resolve()
    if controller.exists():
        raise FileExistsError(controller)
    controller.mkdir(parents=True)
    candidate = args.candidate_root.resolve()
    executor = RemoteExecutor(
        ROOT, controller / "remote-raw", candidate,
        remote={
            "ssh_env_file": str(Path.home() / ".tp_ssh.env"),
            "root": "/root/data-fs/torchport-clip-train-pilot/harness",
            "source_python": "/root/data-fs/APT/.venv-source/bin/python",
            "target_python": "/root/data-fs/APT/.venv-jittor/bin/python",
            "python_config": "",
            "system_libstdcxx": "",
            "nvcc_path": "/usr/local/cuda/bin/nvcc",
            "home": "/root/data-fs/torchport-clip-train-pilot/home",
            "target_path": "/usr/local/cuda/bin:/root/data-fs/APT/.venv-jittor/bin:"
                           "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        },
    )
    report = executor.call("verify", args.task.resolve(), trial=args.trial)
    write_json(controller / "verify-result.json", report)
    print(json.dumps({"controller": str(controller), "passed": report.get("passed"),
                      "failure_code": report.get("failure_code"),
                      "oracle": report.get("oracle")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
