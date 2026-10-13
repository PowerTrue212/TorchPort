"""进程级 Physical Evaluation Cost 聚合。"""

from __future__ import annotations

from typing import Any


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

