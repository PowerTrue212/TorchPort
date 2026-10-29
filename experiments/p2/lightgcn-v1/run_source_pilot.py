"""Build LightGCN source oracle through the real remote verify pipeline.

The target seed intentionally has no migration; its RUN_FAILED is expected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.harness.remote_executor import RemoteExecutor  # noqa: E402
from torchport.contracts.task import load_task  # noqa: E402
from torchport.runtime import write_json  # noqa: E402

BENCH = Path("D:/科研/LightGCN-Benchmark-v1")
REMOTE_CONFIG = Path("D:/科研/CLIP-Benchmark-v1/controller/clip-full-r2/mcp-trial.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--trial", default="lightgcn-source-pilot-r1")
    args = parser.parse_args()
    controller = args.controller.resolve()
    if controller.exists():
        raise FileExistsError(controller)
    remote = dict(json.loads(REMOTE_CONFIG.read_text("utf-8"))["remote"])
    remote.update(root="/root/data-fs/LightGCN-Benchmark-v1/harness",
                  home="/root/data-fs/LightGCN-Benchmark-v1/home",
                  extra_roots=[str(BENCH)])
    controller.mkdir(parents=True)
    bound = json.loads(args.task.resolve().read_text("utf-8"))
    bound["target"]["repo"] = str(args.target.resolve())
    bound_path = controller / "task-bound.json"
    write_json(bound_path, bound)
    if load_task(bound_path)["digest"] != load_task(args.task.resolve())["digest"]:
        raise RuntimeError("binding target changed source oracle identity")
    write_json(controller / "remote-config.json",
               {key: value for key, value in remote.items() if key != "ssh_env_file"})
    executor = RemoteExecutor(ROOT, controller / "remote-raw", args.target.resolve(), remote)
    report = executor.call("verify", bound_path, trial=args.trial)
    write_json(controller / "verify-result.json", report)
    print(json.dumps({"passed": report.get("passed"),
                      "failure_code": report.get("failure_code"),
                      "oracle": report.get("oracle"),
                      "task_digest": report.get("task_digest")}))


if __name__ == "__main__":
    main()
