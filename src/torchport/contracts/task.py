"""TaskSpec 的加载、校验与规范化。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .identity import file_hash, oracle_key, side_identity

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

