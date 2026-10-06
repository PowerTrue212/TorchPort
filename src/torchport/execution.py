"""唯一子进程执行器：cwd、受控环境、超时、进程树终止、隔离缓存、RunManifest。"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# 会污染目标端导入/编译的继承变量；系统与设备变量保留
POLLUTING = ("PYTHONPATH", "PYTHONHOME", "nvcc_path", "JITTOR_HOME", "TORCH_HOME", "cache_path")
SENSITIVE_ENV_MARKERS = (
    "_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTHORIZATION",
    "COOKIE",
    "JWT",
)
SENSITIVE_ENV_NAMES = {"GIT_ASKPASS", "SSH_ASKPASS", "SSH_AUTH_SOCK", "SSH_AGENT_PID"}
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict[str, Any]) -> None:
    """先写临时文件再替换，保证读者看不到半成品。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), "utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_env(side: dict[str, Any], cache: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in POLLUTING and not _sensitive_env_name(key)
    }
    paths = [str(PACKAGE_ROOT), *(str(Path(p).resolve()) for p in side.get("paths", []))]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    # 导入框架前设置隔离缓存；同一 trial/side 内复用
    # Jittor 1.3.x 只认 JITTOR_HOME（缓存在 $JITTOR_HOME/.cache/jittor），cache_path 环境变量不生效
    env["JITTOR_HOME"] = str(cache / "jittor")
    env["TORCH_HOME"] = str(cache / "torch")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name, value in (side.get("env") or {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = str(value)
    if side.get("device", "cpu") == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        # Jittor 发现 nvcc 就会初始化 CUDA；CPU 策略下由执行器显式关闭，而非硬编码
        env["nvcc_path"] = ""
    return env


def _sensitive_env_name(name: str) -> bool:
    upper = name.upper()
    return upper in SENSITIVE_ENV_NAMES or any(marker in upper for marker in SENSITIVE_ENV_MARKERS)


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


def empty_physical_cost() -> dict[str, float | int]:
    return {"executions": 0, "execution_wall_seconds": 0.0, "cuda_execution_seconds": 0.0}


def run_physical_cost(run: dict[str, Any], device: str) -> dict[str, float | int]:
    """进程级物理成本；CUDA 秒表示 CUDA 配置进程的墙钟，不冒充设备活跃时间。"""
    execution = run.get("execution") or {}
    if not execution:
        return empty_physical_cost()
    seconds = float(execution.get("seconds") or 0.0)
    return {
        "executions": 1,
        "execution_wall_seconds": seconds,
        "cuda_execution_seconds": seconds if device == "cuda" else 0.0,
    }


def add_physical_cost(*costs: dict[str, Any] | None) -> dict[str, float | int]:
    result = empty_physical_cost()
    for cost in costs:
        if not cost:
            continue
        result["executions"] += int(cost.get("executions") or 0)
        result["execution_wall_seconds"] += float(cost.get("execution_wall_seconds") or 0.0)
        result["cuda_execution_seconds"] += float(cost.get("cuda_execution_seconds") or 0.0)
    return result
