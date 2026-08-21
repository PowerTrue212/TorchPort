from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_PACKAGES = {
    "jittor": "jittor",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "huggingface-hub": "huggingface_hub",
}
_TOOLS = ("hf", "huggingface-cli", "nvidia-smi", "npu-smi", "git")


def build_doctor_report(*, runtime: bool = False) -> dict[str, Any]:
    # 某些共享文件系统上的发行包元数据扫描非常慢。默认探针只查找模块，
    # 真实版本由 --runtime 子进程导入后报告，避免 doctor 阻塞 SSH 会话。
    packages: dict[str, dict[str, Any]] = {}
    for distribution_name, module_name in _PACKAGES.items():
        spec = importlib.util.find_spec(module_name)
        packages[distribution_name] = {
            "installed": spec is not None,
            "location": spec.origin if spec is not None else None,
        }

    report: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "tools": {name: _which_tool(name) for name in _TOOLS},
        "environment": {
            "JITTOR_HOME": os.environ.get("JITTOR_HOME"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    if runtime:
        report["runtime"] = {
            "jittor": _probe_import("jittor"),
            "torch": _probe_import("torch"),
        }
    return report


def _which_tool(name: str) -> str | None:
    resolved = shutil.which(name)
    if resolved is not None:
        return resolved
    # 通过虚拟环境的 Python 直接调用时，调用者不一定已经激活该环境。
    candidate = Path(sys.executable).parent / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def _probe_import(package: str) -> dict[str, Any]:
    code = (
        f"import {package} as module; "
        "print(getattr(module, '__version__', 'unknown'))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
            # 避免调用者从仅 root 可进入的目录降权执行，导致 nvcc 无法访问当前目录。
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "运行探针超时"}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[-1000:],
        "stderr": completed.stderr.strip()[-3000:],
    }
