"""单端运行编排与 RunManifest 生成。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..comparison import ENVELOPE_V3
from ..runtime import execute, new_run_dir, read_json, write_json

MANIFEST = "manifest.json"
TRAIN_STEP = "train-step"


def _cache(root: Path, trial: str, side: str, python: str | None) -> Path:
    # 以 trial/side/环境 区分；同一 trial 内复用编译缓存，不同 trial 不共享可写缓存
    env = (python or "default").replace("\\", "/").replace("/", "_").replace(":", "")[-40:]
    return root / "cache" / trial / f"{side}-{env}"


def run_side(
    task: dict[str, Any],
    kind: str,
    root: Path,
    *,
    trial: str = "t0",
    mode: str = "normal",
    oracle: Path | None = None,
    performance: dict[str, int] | None = None,
    variant: dict[str, Any] | None = None,
    focus: list[str] | None = None,
    module_oracle: Path | None = None,
    calibrate: bool = False,
    trace_max_bytes: int | None = None,
) -> dict[str, Any]:
    """执行一端，返回 RunManifest；manifest 在全部产物写完后原子提交。

    focus 只用于 trace：只记录这些模块的输出与局部输入。
    calibrate 只用于 train-step 的 source 冻结运行：额外生成 fp64/alt 噪声标定。
    """
    side = task[kind]
    run_dir = new_run_dir(root / "runs", trial, kind)
    train = task["protocol"].get("train") if task["protocol"]["profile"] == TRAIN_STEP else None
    job = {
        "train": train,
        "calibrate": bool(train) and calibrate and kind == "source",
        "focus": focus,
        "trace_max_bytes": trace_max_bytes if mode == "trace" else None,
        # v3：layers 证据中的 output.* 不压缩，供逐元素比较
        "exact_outputs": task["protocol"]["profile"] == ENVELOPE_V3,
        "kind": kind,
        "side": side,
        "inputs": task["inputs"],
        "seed": task["seed"],
        "mode": mode,
        "out_dir": str(run_dir),
        "weights": task["weights"] if kind == "source" else None,
        "oracle": str(oracle) if oracle else None,
        # target 从冻结 oracle 读 state/inputs，从本次 trace-source 读局部调用证据。
        "module_oracle": str(module_oracle) if module_oracle else None,
        "performance": performance,
        "variant": variant,
        # 两端用同一份可推导 buffer 排除清单，完整性检查才对称
        "ignore_state": list(task["protocol"].get("ignore_state") or []),
    }
    write_json(run_dir / "job.json", job)
    record = execute(
        side,
        ["_capture", str(run_dir / "job.json")],
        run_dir=run_dir,
        cache=_cache(root, trial, kind, side.get("python")),
        timeout=float(task["protocol"]["timeout"]),
    )
    capture = run_dir / "capture.json"
    # 路径存在不构成证据：只有本次进程成功且写出 capture.json 才算产物
    ok = record["ok"] and capture.is_file()
    artifacts = {"outputs": str(run_dir / "outputs.npz")} if ok else {}
    if ok and (run_dir / "trace.npz").is_file():
        artifacts["trace"] = str(run_dir / "trace.npz")
    if ok and (run_dir / "layers.npz").is_file():
        artifacts["layers"] = str(run_dir / "layers.npz")
    if ok and (run_dir / "module_outputs.npz").is_file():
        artifacts["module_outputs"] = str(run_dir / "module_outputs.npz")
    for name in ("module_fp64", "module_alt", "module_repeat", "train_fp64", "train_alt"):
        if ok and (run_dir / f"{name}.npz").is_file():
            artifacts[name] = str(run_dir / f"{name}.npz")
    if ok and (run_dir / "module_inputs.npz").is_file():
        artifacts["module_inputs"] = str(run_dir / "module_inputs.npz")
    if ok and (run_dir / "module_config.json").is_file():
        artifacts["module_config"] = str(run_dir / "module_config.json")
    if ok and (run_dir / "module_shapes.json").is_file():
        artifacts["module_shapes"] = str(run_dir / "module_shapes.json")
    if ok and (run_dir / "module_calls.json").is_file():
        artifacts["module_calls"] = str(run_dir / "module_calls.json")
    if ok and (run_dir / "isolated.npz").is_file():
        artifacts["isolated"] = str(run_dir / "isolated.npz")
    if ok and (run_dir / "isolated_repeat.npz").is_file():
        artifacts["isolated_repeat"] = str(run_dir / "isolated_repeat.npz")
    manifest = {
        "kind": "torchport.run",
        "run_id": run_dir.name,
        "trial_id": trial,
        "side": kind,
        "mode": mode,
        "task_id": task["task_id"],
        "task_digest": task["digest"],
        # TaskSpec 已冻结本端内容身份；repeat 不再重复遍历整个仓库。
        "side_identity": task[f"{kind}_identity"],
        "oracle": str(oracle) if oracle else None,
        "execution": record,
        "ok": ok,
        "dir": str(run_dir),
        "capture": read_json(capture) if ok else None,
        "artifacts": artifacts,
        # 契约错误只在运行失败时出现；内容是冻结枚举 {code, message}
        "contract": (
            read_json(run_dir / "contract.json")
            if not ok and (run_dir / "contract.json").is_file() else None
        ),
    }
    write_json(run_dir / MANIFEST, manifest)
    return manifest

