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


def _train_protocol(protocol: dict[str, Any], inputs: list[Any]) -> None:
    """train-step 需要两端同名的 optimizer 工厂与单步函数；输入第 0 维是步号。"""
    train = protocol.get("train")
    if not isinstance(train, dict):
        raise ValueError("profile=train-step 需要 protocol.train 对象")
    for key in ("optimizer", "step"):
        if not isinstance(train.get(key), str) or not train[key]:
            raise ValueError(f"protocol.train.{key} 必须是入口中的函数名")
    steps = {
        int(item["shape"][0])
        for item in inputs
        if isinstance(item, dict) and item.get("shape")
    }
    if len(steps) != 1 or min(steps) < 1:
        raise ValueError("train-step 的每个输入第 0 维必须是相同的正步数")
    normalized = {"optimizer": train["optimizer"], "step": train["step"]}
    if "observe" in train:
        observed = train["observe"]
        if (
            not isinstance(observed, list)
            or not observed
            or any(not isinstance(name, str) or not name for name in observed)
            or len(observed) != len(set(observed))
        ):
            raise ValueError("protocol.train.observe 必须是非空且无重复的参数名列表")
        normalized["observe"] = observed
    if "sketch_dim" in train:
        # 全参数更新量的 CountSketch 维度；与 observe 二选一
        dim = train["sketch_dim"]
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError("protocol.train.sketch_dim 必须是正整数")
        if "observe" in normalized:
            raise ValueError("protocol.train.sketch_dim 与 observe 不能同时使用")
        normalized["sketch_dim"] = dim
    protocol["train"] = normalized


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
    if protocol["profile"] == "train-step":
        _train_protocol(protocol, inputs)
    elif protocol.get("train") is not None:
        raise ValueError("protocol.train 只用于 profile=train-step")
    envelope = protocol.get("train_envelope")
    baseline = protocol.get("train_noise_baseline")
    if envelope is not None or baseline is not None:
        if protocol["profile"] != "train-step" or envelope != "train-repeat-envelope-v1-pilot":
            raise ValueError("固定参考重复性包络只用于 train-step pilot")
        if not isinstance(baseline, str) or not baseline:
            raise ValueError("train-repeat-envelope-v1-pilot 缺少 train_noise_baseline")
        baseline_path = Path(baseline).expanduser()
        baseline_path = (baseline_path if baseline_path.is_absolute() else base / baseline_path).resolve()
        if baseline_path.suffix.lower() != ".npz" or not baseline_path.is_file():
            raise ValueError(f"固定参考重复性文件不存在：{baseline_path}")
        baseline_hash = file_hash(baseline_path)
        if protocol.get("train_noise_baseline_sha256") != baseline_hash:
            raise ValueError("固定参考重复性文件 SHA256 不匹配")
        protocol["train_noise_baseline"] = str(baseline_path)
    elif protocol.get("train_noise_baseline_sha256") is not None:
        raise ValueError("train_noise_baseline_sha256 缺少对应标定文件")
    if "ignore_state" in protocol:
        # 可由形状推出、不参与训练的 buffer（如 embeddings.position_ids），两端都不计入完整性
        ignored = protocol["ignore_state"]
        if (
            not isinstance(ignored, list)
            or not ignored
            or any(not isinstance(name, str) or not name for name in ignored)
            or len(ignored) != len(set(ignored))
        ):
            raise ValueError("protocol.ignore_state 必须是非空且无重复的状态名列表")
        protocol["ignore_state"] = sorted(ignored)
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

