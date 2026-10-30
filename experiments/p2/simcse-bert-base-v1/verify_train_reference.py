"""Run hidden `verify` for a controller-only training reference on the V100.

Same remote environment as the r4 trials, except that the harness root and HOME
can be moved off the full root partition. The oracle identity does not depend on
these paths; a new root only rebuilds the same source oracle there.
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

BENCH = Path("D:/科研/SimCSE-Benchmark-v1")
REMOTE_CONFIG = BENCH / "controller/simcse-train-full-r4/mcp-trial.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-home", required=True)
    # 远端工作目录按 TaskSpec 路径哈希命名；多个候选串行共用一个路径即可复用同一份 oracle
    parser.add_argument("--shared-task", type=Path)
    # trace 只采集诊断证据（B_ℓ 用），不签发验收
    parser.add_argument("--tool", choices=("verify", "trace"), default="verify")
    args = parser.parse_args()
    controller = args.controller.resolve()
    if controller.exists():
        raise FileExistsError(controller)
    remote = dict(json.loads(REMOTE_CONFIG.read_text("utf-8"))["remote"])
    remote.update(root=args.remote_root, home=args.remote_home,
                  extra_roots=[str(BENCH)])
    controller.mkdir(parents=True)
    # RemoteExecutor syncs candidate_root, but the runner executes task.target.repo.
    # Bind that field explicitly in a new controller-only task; never edit the
    # original frozen task or silently re-run the reference under a new label.
    bound_task = json.loads(args.task.resolve().read_text("utf-8"))
    bound_task["target"]["repo"] = str(args.candidate_root.resolve())
    bound_path = controller / "task-bound.json"
    write_json(bound_path, bound_task)
    if load_task(bound_path)["digest"] != load_task(args.task.resolve())["digest"]:
        raise RuntimeError("绑定 target 后 source oracle 身份发生变化")
    if args.shared_task is not None:
        args.shared_task.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.shared_task, bound_task)
        bound_path = args.shared_task.resolve()
    write_json(controller / "remote-config.json",
               {key: value for key, value in remote.items() if key != "ssh_env_file"})
    executor = RemoteExecutor(ROOT, controller / "remote-raw",
                              args.candidate_root.resolve(), remote)
    report = executor.call(args.tool, bound_path, trial=args.trial)
    write_json(controller / f"{args.tool}-result.json", report)
    print(json.dumps({
        "controller": str(controller),
        "passed": report.get("passed"),
        "failure_code": report.get("failure_code"),
        "oracle_path": report.get("oracle"),
        "task_digest": report.get("task_digest"),
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
