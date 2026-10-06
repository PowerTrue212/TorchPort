"""TaskSpec：加载、校验、规范化冻结的迁移任务。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
SIDES = ("source", "target")
DEFAULT_PROTOCOL = {
    "profile": "fp32-inference",
    "repeats": 3,
    "timeout": 600,
    "check_dtype": True,
    "allow_scalar_singleton": False,
}


def _side(name: str, raw: dict[str, Any], base: Path) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"task.{name} 必须是对象")
    missing = [key for key in ("repo", "entry", "model") if not raw.get(key)]
    if missing:
        raise ValueError(f"task.{name} 缺少字段：{', '.join(missing)}")
    repo = Path(raw["repo"]).expanduser()
    return {
        "repo": str((repo if repo.is_absolute() else base / repo).resolve()),
        "entry": str(raw["entry"]),
        "model": str(raw["model"]),
        "args": list(raw.get("args") or []),
        "kwargs": dict(raw.get("kwargs") or {}),
        "python": raw.get("python"),
        "device": raw.get("device", "cpu"),
        "paths": [
            str((path if path.is_absolute() else base / path).resolve())
            for item in raw.get("paths") or []
            for path in [Path(item).expanduser()]
        ],
        "env": dict(raw.get("env") or {}),
    }


def _input(raw: Any, base: Path) -> dict[str, Any]:
    if not isinstance(raw, dict) or not raw.get("shape"):
        raise ValueError("task.inputs 每项必须是包含 shape 的对象")
    value = dict(raw)
    if raw.get("path"):
        path = Path(raw["path"]).expanduser()
        path = (path if path.is_absolute() else base / path).resolve()
        if path.suffix.lower() != ".npy" or not path.is_file():
            raise ValueError(f"固定输入必须是存在的 .npy 文件：{path}")
        value["path"] = str(path)
        value["sha256"] = file_hash(path)
    return value


def load_task(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"task 文件必须是对象：{path}")
    return normalize(raw, path.parent)


def normalize(raw: dict[str, Any], base: Path) -> dict[str, Any]:
    if raw.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(f"不支持的 task schema_version：{raw.get('schema_version')}")
    if not raw.get("task_id"):
        raise ValueError("task 缺少 task_id")
    inputs = raw.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("task 缺少 inputs")
    protocol = {**DEFAULT_PROTOCOL, **(raw.get("protocol") or {})}
    if int(protocol["repeats"]) < 1 or int(protocol["timeout"]) <= 0:
        raise ValueError("protocol.repeats 和 protocol.timeout 必须为正数")
    weights = raw.get("weights")
    if weights:
        weights = Path(weights).expanduser()
        weights = str((weights if weights.is_absolute() else base / weights).resolve())
    task = {
        "schema_version": SCHEMA_VERSION,
        "task_id": str(raw["task_id"]),
        "source": _side("source", raw.get("source"), base),
        "target": _side("target", raw.get("target"), base),
        "inputs": [_input(item, base) for item in inputs],
        "weights": weights,
        "seed": int(raw.get("seed", 7)),
        "protocol": protocol,
    }
    if weights and not Path(weights).is_file():
        raise ValueError(f"task.weights 不存在：{weights}")
    task["weights_sha"] = file_hash(weights) if weights else None
    task["source_identity"] = side_identity(task["source"])
    task["target_identity"] = side_identity(task["target"])
    task["digest"] = oracle_key(task)
    return task


def digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(data).hexdigest()


def oracle_key(task: dict[str, Any]) -> str:
    """oracle 身份：source 代码内容/配置/解释器、checkpoint 内容、输入与协议；target 不参与。"""
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
            "protocol": task["protocol"],
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
