#!/usr/bin/env python3
"""Run hidden official verify after a DSH session and save the full report evaluator-side."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = KIT.parents[2]
RESULTS_ROOT = KIT / "results"
RUNNER = PROJECT_ROOT / "scripts" / "remote" / "dsh_torchport.py"


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DSH 会话结束后运行隐藏 verify；完整 JSON 只写入 heldout 结果目录。"
    )
    parser.add_argument("task", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trial", required=True)
    parser.add_argument(
        "--session-ended",
        action="store_true",
        help="确认对应 DSH 会话已经停止，后续输出不会返回给 Agent。",
    )
    args = parser.parse_args()
    if not args.session_ended:
        parser.error("必须在 DSH 会话停止后传入 --session-ended")

    task_path = args.task.resolve()
    output_path = args.output.resolve()
    results_root = RESULTS_ROOT.resolve()
    if not _inside(output_path, results_root):
        parser.error(f"output 必须位于评测方结果目录：{results_root}")
    if output_path.suffix.lower() != ".json":
        parser.error("output 必须是 .json 文件")

    task = json.loads(task_path.read_text(encoding="utf-8"))
    target_repo = Path(task["target"]["repo"]).resolve()
    if _inside(output_path, target_repo):
        parser.error("禁止把正式验收结果写入 Agent 工作区")

    env = os.environ.copy()
    env["TP_PROJECT_ROOT"] = str(PROJECT_ROOT)
    extra_roots = [item for item in env.get("TP_EXTRA_ROOTS", "").split(os.pathsep) if item]
    if str(target_repo.parent) not in extra_roots:
        extra_roots.append(str(target_repo.parent))
    env["TP_EXTRA_ROOTS"] = os.pathsep.join(extra_roots)
    # 评测方需要保存完整原始报告；该进程不属于已经停止的 DSH 会话。
    env["TP_FEEDBACK"] = "localized"

    command = [
        sys.executable,
        str(RUNNER),
        "verify",
        str(task_path),
        "--root",
        "official",
        "--trial",
        args.trial,
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
    if completed.stderr:
        output_path.with_suffix(".stderr.txt").write_text(
            completed.stderr, encoding="utf-8", newline="\n"
        )
    if completed.returncode:
        print(
            json.dumps(
                {
                    "completed": False,
                    "exit_code": completed.returncode,
                    "report": str(output_path),
                },
                ensure_ascii=False,
            )
        )
        return completed.returncode

    report = json.loads(completed.stdout)
    print(
        json.dumps(
            {
                "completed": True,
                "passed": report.get("passed"),
                "e1_status": report.get("e1_status"),
                "failure_code": report.get("failure_code"),
                "report": str(output_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
