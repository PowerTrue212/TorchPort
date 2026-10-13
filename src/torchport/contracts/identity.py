"""Task、代码树、权重和 oracle 的内容身份。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(data).hexdigest()


# 证据采集语义的版本。capture 的改动若会改变已记录的数值（例如 2026-10-09 修复 hook 记录与
# 原地运算共享内存），必须递增，使旧 oracle/envelope 失效重建。
CAPTURE_VERSION = 2
# 只影响判定、不影响 source 证据的协议参数；调整它们不应使 oracle/envelope 失效重建。
# 实际取值仍完整写入每份 verify 报告。
JUDGE_KEYS = ("pass_k", "floor_k", "noise_floor", "noise_probe")


def oracle_key(task: dict[str, Any]) -> str:
    """oracle 身份：source 代码内容/配置/解释器、checkpoint 内容、输入、协议与采集版本；target 不参与。"""
    portable_inputs = [
        {key: value for key, value in spec.items() if key != "path"}
        for spec in task["inputs"]
    ]
    return digest(
        {
            "source_identity": task["source_identity"],
            "inputs": portable_inputs,
            "weights_sha": task["weights_sha"],
            "seed": task["seed"],
            "protocol": {k: v for k, v in task["protocol"].items() if k not in JUDGE_KEYS},
            "capture_version": CAPTURE_VERSION,
        }
    )


CODE_SUFFIXES = {
    ".py", ".pyx", ".pyi", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".txt",
    ".cu", ".cc", ".cpp", ".h",
}
SKIP_DIRS = {
    ".cache",
    ".eggs",
    ".git",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".torchport",
    ".tox",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "checkpoints",
    "datasets",
    "dist",
    "logs",
    "node_modules",
    "outputs",
    "results",
    "venv",
    "wandb",
    "weights",
}


def tree_hash(root: str | Path) -> str:
    """按相对路径 + 内容哈希摘要代码/配置文件；权重另由 weights_sha 覆盖。"""
    root = Path(root).resolve()
    sha = hashlib.sha256()
    for directory, names, files in os.walk(root, followlinks=False):
        # rglob 会先遍历再过滤；这里直接剪枝，避免进入虚拟环境、数据和历史产物目录。
        names[:] = sorted(name for name in names if name not in SKIP_DIRS)
        base = Path(directory)
        for name in sorted(files):
            path = base / name
            if path.suffix.lower() not in CODE_SUFFIXES or not path.is_file():
                continue
            rel = path.relative_to(root)
            sha.update(rel.as_posix().encode() + b"\0" + file_hash(path).encode())
    return sha.hexdigest()


def side_identity(side: dict[str, Any]) -> str:
    """一端的内容身份：代码摘要 + 构造配置 + 解释器/环境；与目录位置无关。"""
    config = {key: value for key, value in side.items() if key != "repo"}
    return digest({**config, "code": tree_hash(side["repo"])})


def file_hash(path: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()

