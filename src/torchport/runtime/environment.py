"""子进程环境清理、显式覆盖与缓存隔离。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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

