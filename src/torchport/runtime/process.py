"""唯一子进程执行器与独立运行目录。"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .environment import build_env


def _kill_tree(process: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGKILL)


def execute(
    side: dict[str, Any],
    args: list[str],
    *,
    run_dir: Path,
    cache: Path,
    timeout: float,
    module: str | None = "torchport",
) -> dict[str, Any]:
    """在 side.repo 下运行 `python -m <module> <args>`（module=None 时直接运行 python <args>），
    返回执行记录（不抛异常）。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    python = str(side.get("python") or sys.executable)
    command = [python, "-m", module, *args] if module else [python, *args]
    started = time.time()
    record: dict[str, Any] = {
        "command": command,
        "cwd": side["repo"],
        "cache": str(cache),
        "started": started,
        "timeout": False,
        "exit_code": None,
    }
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as out, (
        run_dir / "stderr.log"
    ).open("w", encoding="utf-8") as err:
        process = subprocess.Popen(
            command,
            cwd=side["repo"],
            env=build_env(side, cache),
            stdout=out,
            stderr=err,
            text=True,
            start_new_session=os.name != "nt",
        )
        try:
            record["exit_code"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(process)
            process.wait()
            record["timeout"] = True
    record["finished"] = time.time()
    record["seconds"] = record["finished"] - started
    record["ok"] = record["exit_code"] == 0 and not record["timeout"]
    if not record["ok"]:
        record["stderr_tail"] = (run_dir / "stderr.log").read_text("utf-8", "replace")[-3000:]
    return record


def new_run_dir(root: Path, trial: str, side: str) -> Path:
    """每次运行一个新目录：失败不会被误读为上一轮产物。"""
    run_id = f"{side}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    path = root / trial / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path

