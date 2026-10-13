"""Agent 可见的公开 TaskSpec 检查。"""

from __future__ import annotations

from typing import Any


def inspect_task(task: dict[str, Any]) -> dict[str, Any]:
    """返回公开任务契约，不读取或生成 oracle。"""
    def side(name: str) -> dict[str, Any]:
        value = task[name]
        return {
            "repo": value["repo"],
            "entry": value["entry"],
            "model": value["model"],
            "args": value["args"],
            "kwargs": value["kwargs"],
            "device": value["device"],
            "python": value.get("python"),
        }

    return {
        "kind": "torchport.inspect",
        "task_id": task["task_id"],
        "task_digest": task["digest"],
        "source": side("source"),
        "target": side("target"),
        "inputs": task["inputs"],
        "protocol": task["protocol"],
        "checkpoint": {"provided": bool(task.get("weights")), "sha256": task.get("weights_sha")},
        "candidate_identity": task["target_identity"],
    }

