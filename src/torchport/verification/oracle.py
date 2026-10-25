"""Reference oracle 的构建、加载与内容校验。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import file_hash
from ..runtime import (
    add_physical_cost,
    empty_physical_cost,
    read_json,
    run_physical_cost,
    write_json,
)
from .runner import run_side

ORACLE_KIND = "torchport.oracle"


def build_oracle(task: dict[str, Any], root: Path, *, rebuild: bool = False) -> dict[str, Any]:
    """source 运行 repeats 次，输出须逐位稳定，才冻结为 oracle；协议摘要不变时复用。"""
    path = root / "oracle" / task["digest"][:16]
    meta_path = path / "oracle.json"
    if meta_path.is_file() and not rebuild:
        meta = read_json(meta_path)
        if _oracle_valid(meta, task, path):
            return {
                **meta,
                "oracle_reused": True,
                "physical_cost": empty_physical_cost(),
            }
    # train-step 的 fp64/alt 噪声标定只在首次运行生成；后续运行只用于逐位稳定性检查。
    runs = [
        run_side(task, "source", root, trial=f"oracle-{i}", calibrate=i == 0)
        for i in range(int(task["protocol"]["repeats"]))
    ]
    failed = next((run for run in runs if not run["ok"]), None)
    physical = add_physical_cost(
        *(run_physical_cost(run, task["source"]["device"]) for run in runs)
    )
    if failed:
        return {
            "kind": ORACLE_KIND,
            "ok": False,
            "reason": "SOURCE_RUN_FAILED",
            "run": failed,
            "physical_cost": physical,
        }
    hashes = {run["capture"]["hashes"]["output"] for run in runs}
    if len(hashes) != 1:
        return {
            "kind": ORACLE_KIND,
            "ok": False,
            "reason": "ORACLE_UNSTABLE",
            "runs": runs,
            "physical_cost": physical,
        }
    first = Path(runs[0]["dir"])
    names = ["inputs.npz", "outputs.npz", "state.npz"]
    if (runs[0]["capture"] or {}).get("train"):
        calibration = ["train_fp64.npz", "train_alt.npz"]
        if not all((first / name).is_file() for name in calibration):
            # 缺少噪声标定时不能冻结 oracle，否则 verify 会失去判定依据
            return {
                "kind": ORACLE_KIND,
                "ok": False,
                "reason": "SOURCE_CALIBRATION_MISSING",
                "runs": runs,
                "physical_cost": physical,
            }
        names.extend(calibration)
    path.mkdir(parents=True, exist_ok=True)
    for name in names:
        (path / name).write_bytes((first / name).read_bytes())
    meta = {
        "kind": ORACLE_KIND,
        "ok": True,
        "task_id": task["task_id"],
        "task_digest": task["digest"],
        "dir": str(path),
        "capture": runs[0]["capture"],
        "files": {name: file_hash(path / name) for name in names},
        "source_runs": [run["run_id"] for run in runs],
        "build_physical_cost": physical,
    }
    write_json(meta_path, meta)
    return {**meta, "oracle_reused": False, "physical_cost": physical}


def load_oracle(task: dict[str, Any], root: Path) -> dict[str, Any]:
    """只读加载已冻结的 oracle；缺失或身份不符时返回不可用，不启动模型。"""
    path = root / "oracle" / task["digest"][:16]
    meta_path = path / "oracle.json"
    if not meta_path.is_file():
        return {"kind": ORACLE_KIND, "ok": False, "reason": "ORACLE_MISSING"}
    meta = read_json(meta_path)
    if not _oracle_valid(meta, task, path):
        return {"kind": ORACLE_KIND, "ok": False, "reason": "ORACLE_INVALID"}
    return {**meta, "oracle_reused": True, "physical_cost": empty_physical_cost()}


def _oracle_valid(meta: dict[str, Any], task: dict[str, Any], path: Path) -> bool:
    # 按内容与协议身份校验，不按目录名或时间戳猜测
    if not meta.get("ok") or meta.get("task_digest") != task["digest"]:
        return False
    files = meta.get("files") or {}
    return all((path / n).is_file() and file_hash(path / n) == h for n, h in files.items()) and bool(
        files
    )

