"""单端采集主流程。

在子进程内运行（`python -m torchport _capture <side.json>`），只导入本端框架。
"""

from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import file_hash
from ..runtime import read_json, write_json
from .inputs import input_key, make_data, pack_values
from .model import (
    _from_numpy,
    _fw,
    _synchronize,
    import_paths,
    load_checkpoint,
    load_model,
    load_scope,
    load_state,
    save_state,
    to_arrays,
)
from .sketch import _is_output_name, module_config, sketch_layers
from .tracing import (
    OpTrace,
    _mark_leaves,
    add_hooks,
    calibrate_source_modules,
    func_scopes,
    run_isolated_modules,
)
from .training import calibrate_steps, enable_determinism, run_steps
from .variants import _unbatch, _variant_inputs, apply_variant

MODES = ("normal", "trace", "layers")


def _digest(values: dict[str, np.ndarray]) -> str:
    import hashlib

    sha = hashlib.sha256()
    for name in sorted(values):
        data = np.ascontiguousarray(values[name])
        sha.update(name.encode() + str(data.dtype).encode() + str(data.shape).encode())
        sha.update(data.tobytes())
    return sha.hexdigest()


def _check_bridge(kind: str) -> None:
    if kind != "target":
        return
    import jittor as jt

    expected = np.asarray([-1.25, 0.0, 2.5], dtype=np.float32)
    if not np.array_equal(expected, np.asarray(jt.array(expected).numpy())):
        raise ValueError(f"Jittor/NumPy 数组桥接不兼容：NumPy {np.__version__}；请使用 NumPy < 2")


def _framework(kind: str, seed: int, device: str) -> dict[str, Any]:
    """设置随机种子与设备，并记录实际生效的设备与缓存路径（不只记录请求值）。"""
    fw = _fw(kind)
    if kind == "source":
        fw.manual_seed(seed)
        if device == "cuda" and not fw.cuda.is_available():
            raise ValueError("task 要求 cuda，但 PyTorch 未检测到可用 GPU")
        return {
            "name": "pytorch",
            "version": fw.__version__,
            "device": device,
            "cuda": fw.version.cuda,
            "device_name": fw.cuda.get_device_name(0) if device == "cuda" else "cpu",
        }
    fw.set_global_seed(seed)
    if device == "cuda":
        if not fw.has_cuda:
            raise ValueError("task 要求 cuda，但 Jittor 未检测到可用 GPU")
        fw.flags.use_cuda = 1
    else:
        fw.flags.use_cuda = 0
    from jittor import compiler

    return {
        "name": "jittor",
        "version": fw.__version__,
        "device": "cuda" if fw.flags.use_cuda else "cpu",
        "cache_path": compiler.cache_path,  # 实际生效路径，供隔离核验
    }


def _call_model(model: Any, args: list[Any], kwargs: dict[str, Any], kind: str) -> Any:
    if kind == "source":
        with _fw(kind).no_grad():
            return model(*args, **kwargs)
    return model(*args, **kwargs)


def _train_steps(inputs: list[dict[str, Any]]) -> int:
    steps = {int((spec.get("shape") or [0])[0]) for spec in inputs}
    if len(steps) != 1 or min(steps) < 1:
        raise ValueError("train-step 的每个输入第 0 维必须是相同的正步数")
    return steps.pop()


def _capture_train(
    job: dict[str, Any],
    model: Any,
    scope: dict[str, Any],
    arrays: dict[str, np.ndarray],
    state: dict[str, Any],
    state_path: Path,
    checkpoint: Any,
    framework: dict[str, Any],
    device: str,
    fresh_source: bool,
) -> dict[str, Any]:
    """训练步采集：只在冻结 source 首次运行时额外生成 fp64/alt 噪声标定。"""
    kind, out, train = job["kind"], Path(job["out_dir"]), job["train"]
    steps = _train_steps(job["inputs"])
    calibrate = kind == "source" and fresh_source and bool(job.get("calibrate"))
    initial = deepcopy(model) if calibrate else None
    outputs, info = run_steps(model, kind, device, scope, train, arrays, job["inputs"], steps)
    if kind == "source" and fresh_source:
        np.savez(out / "inputs.npz", **arrays)
    np.savez(out / "outputs.npz", **arrays, **outputs)
    if calibrate:
        fp64, alt = calibrate_steps(
            initial, device, scope, train, arrays, job["inputs"], steps, job["seed"]
        )
        np.savez(out / "train_fp64.npz", **fp64)
        np.savez(out / "train_alt.npz", **alt)
    report = {
        "kind": kind,
        "mode": "normal",
        "framework": framework,
        "numpy": np.__version__,
        "weights": state["total"],
        "loaded": state["loaded"],
        "state": state,
        "checkpoint": checkpoint,
        "outputs": sorted(outputs),
        "hashes": {
            "input": _digest(arrays),
            "output": _digest(outputs),
            "state": file_hash(state_path) if kind == "source" else None,
        },
        "train": {**info, "calibrated": calibrate},
    }
    write_json(out / "capture.json", report)
    return report


def capture(job: dict[str, Any]) -> dict[str, Any]:
    """job: kind/side/inputs/seed/mode/out_dir，以及 oracle 或 source weights。

    target 必须从 oracle 读 state/inputs；聚焦 source trace 也可直接复用同一份
    冻结证据，避免在 run 目录重复写权重。
    """
    capture_started = time.time()
    kind, side, mode = job["kind"], job["side"], job.get("mode", "normal")
    if mode not in MODES:
        raise ValueError(f"未知采集模式：{mode}")
    out = Path(job["out_dir"])
    root = Path(side["repo"])
    device = side.get("device", "cpu")
    if job.get("train"):
        if mode != "normal" or job.get("performance") or job.get("variant"):
            raise ValueError("train-step 只支持 mode=normal，不支持性能测量和变体")
        # 必须在任何 CUDA 工作之前开启
        enable_determinism(kind)
    framework = _framework(kind, job["seed"], device)
    _check_bridge(kind)
    original = list(sys.path)
    sys.path[:0] = [str(path) for path in import_paths(root)]
    try:
        scope = load_scope(root, side["entry"])
        model = load_model(scope, side["model"], kind, side["args"], side["kwargs"])
    finally:
        sys.path[:] = original
    # train-step 不进 eval：Jittor 的 eval() 会对全部参数 stop_grad，train() 不能可靠恢复，
    # 冻结状态必须保持模型构造时的样子，否则可训练参数集合被采集流程本身改写。
    if not job.get("train"):
        model.eval()
    if kind == "source" and device == "cuda":
        model.cuda()

    oracle = Path(job["oracle"]) if job.get("oracle") else None
    ignore = tuple(job.get("ignore_state") or ())
    state_path = out / "state.npz"
    if kind == "source":
        if oracle is not None:
            with np.load(oracle / "inputs.npz", allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            state_path = oracle / "state.npz"
            state = load_state(model, state_path, kind="source", ignore=ignore)
            oracle_meta = read_json(oracle / "oracle.json")
            checkpoint = (oracle_meta.get("capture") or {}).get("checkpoint")
        else:
            weights = job.get("weights")
            checkpoint = (
                load_checkpoint(model, Path(weights), ignore)
                if weights
                else {"requested": False, "complete": True}
            )
            arrays = make_data(job["inputs"], job["seed"])
            total = save_state(model, state_path, ignore)
            state = {"total": total, "loaded": total, "missing": [], "value_mismatch": []}
    else:
        if oracle is None:
            raise ValueError("target 端缺少 oracle")
        with np.load(oracle / "inputs.npz", allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        state = load_state(model, oracle / "state.npz", ignore=ignore)
        checkpoint = None

    if job.get("train"):
        return _capture_train(
            job,
            model,
            scope,
            arrays,
            state,
            state_path,
            checkpoint,
            framework,
            device,
            fresh_source=kind == "source" and oracle is None,
        )

    if job.get("variant") and kind != "source":
        raise ValueError("合法变体只施加在 source 端")
    variant = apply_variant(model, job.get("variant")) if kind == "source" else None
    if variant and variant["device"] == "cpu":
        device = "cpu"
    tensors = [_from_numpy(arrays[input_key(i, s)], kind) for i, s in enumerate(job["inputs"])]
    if kind == "source" and device == "cuda":
        tensors = [tensor.cuda() for tensor in tensors]
    tensors = _variant_inputs(tensors, variant)
    args, kwargs = pack_values(tensors, job["inputs"])
    focus = set(job["focus"]) if job.get("focus") else None
    trace_values: dict[str, np.ndarray] = {}
    module_values: dict[str, np.ndarray] = {}
    module_calls: list[dict[str, Any]] = []
    module_inputs: dict[str, np.ndarray] = {}
    layer_values: dict[str, np.ndarray] = {}
    # source 参考运行顺带记录各模块输入形状，供 N_env 按真实归约规模标定
    module_shapes: dict[str, list[int]] | None = (
        {} if mode == "layers" and kind == "source" and not job.get("variant") else None
    )
    if mode == "layers":
        # E1 只要整网前向中每个模块的真实输出；不做算子包装和模块隔离。
        handles = add_hooks(model, kind, layer_values, shapes=module_shapes)
    elif mode == "trace":
        handles = add_hooks(
            model,
            kind,
            trace_values,
            root=root,
            probe_values=module_values,
            probe_calls=module_calls,
            probe_inputs=module_inputs,
            focus=focus,
        )
    else:
        handles = []
    # 聚焦 trace 不包装函数式算子：op.* 无法归属到模块，全量记录正是产物膨胀的来源之一
    ops = (
        OpTrace(kind, trace_values, [scope, *func_scopes(model, kind)])
        if mode == "trace" and focus is None
        else None
    )
    if ops:
        ops.start()
    try:
        result = _call_model(model, args, kwargs, kind)
        _synchronize(kind, device, result)
        first_output_at = time.time()
    finally:
        if ops:
            ops.stop()
        for handle in handles:
            handle.remove()
    # Jittor 的 register_forward_hook 不返回句柄，hook 会一直留在模块上；
    # 先冻结整网前向的 trace，避免后续隔离/重复运行继续写入。
    trace_values = dict(trace_values)
    layer_values = dict(layer_values)
    isolated_values: dict[str, np.ndarray] = {}
    isolated_repeats: dict[str, np.ndarray] = {}
    isolated = None
    if mode == "trace" and kind == "target":
        isolated_values, isolated_repeats, isolated = run_isolated_modules(
            model, Path(job.get("module_oracle") or job["oracle"]), kind, device
        )
    performance = None
    if requested := job.get("performance"):
        if mode != "normal":
            raise ValueError("性能测量只允许 mode=normal")
        warmup = int(requested.get("warmup", 10))
        iterations = int(requested.get("iterations", 30))
        if warmup < 0 or iterations < 1:
            raise ValueError("performance warmup 必须非负且 iterations 必须为正数")
        for _ in range(warmup):
            result = _call_model(model, args, kwargs, kind)
        _synchronize(kind, device, result)
        latency_ms = []
        for _ in range(iterations):
            started = time.perf_counter()
            result = _call_model(model, args, kwargs, kind)
            _synchronize(kind, device, result)
            latency_ms.append((time.perf_counter() - started) * 1000)
        ordered = sorted(latency_ms)
        median = float(np.median(ordered))
        p90 = float(ordered[max(0, int(np.ceil(0.9 * len(ordered))) - 1)])
        shape = list(job["inputs"][0].get("shape") or [])
        batch = int(shape[0]) if shape else 1
        performance = {
            "warmup": warmup,
            "iterations": iterations,
            "batch_size": batch,
            "first_output_at": first_output_at,
            "capture_started_at": capture_started,
            "latency_ms": latency_ms,
            "steady_latency_ms": {"median": median, "p90": p90},
            "throughput_per_second": batch / (median / 1000),
        }
    outputs = to_arrays(result, kind, "output")
    if not outputs:
        raise ValueError("模型输出中没有张量")
    if kind == "source" and oracle is None:
        np.savez(out / "inputs.npz", **arrays)
    np.savez(out / "outputs.npz", **arrays, **outputs)
    if trace_values:
        # 聚焦 trace 中的容器模块仍可输出数千万元素；大张量只保留
        # CountSketch 供 L2 证据使用，小张量仍逐元素保留。全量 trace 不改。
        stored_trace = sketch_layers(trace_values) if focus is not None else trace_values
        np.savez(out / "trace.npz", **stored_trace)
    if mode == "layers":
        # 最终输出也进入 E1：中间层可以因命名不同而只报告，output.* 必须参与判定。
        batch = variant["batch"] if variant else 1
        batch_mode = variant.get("batch_mode", "copy") if variant else "copy"
        layers = _unbatch({**layer_values, **outputs}, batch, batch_mode)
        # v3 的输出逐元素比较：output.* 不压缩
        exact = {name for name in layers if job.get("exact_outputs") and _is_output_name(name)}
        np.savez(out / "layers.npz", **sketch_layers(layers, keep=exact))
    calibration = None
    if kind == "source" and mode == "trace":
        _mark_leaves(module_calls)
        fp64_values, alt_values, repeat_values, calibration = calibrate_source_modules(
            model, module_calls, module_inputs, device
        )
        np.savez(out / "module_fp64.npz", **fp64_values)
        np.savez(out / "module_alt.npz", **alt_values)
        np.savez(out / "module_repeat.npz", **repeat_values)
        np.savez(out / "module_outputs.npz", **module_values)
        np.savez(out / "module_inputs.npz", **module_inputs)
        (out / "module_calls.json").write_text(
            json.dumps(module_calls, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if isolated_values:
        np.savez(out / "isolated.npz", **isolated_values)
    if isolated_repeats:
        np.savez(out / "isolated_repeat.npz", **isolated_repeats)
    report = {
        "kind": kind,
        "mode": mode,
        "framework": framework,
        "numpy": np.__version__,
        "weights": state["total"],
        "loaded": state["loaded"],
        "state": state,
        "checkpoint": checkpoint,
        "outputs": sorted(outputs),
        "hashes": {
            "input": _digest(arrays),
            "output": _digest(outputs),
            # 聚焦 source trace 直接使用冻结 oracle；哈希指向真实加载的文件，
            # 不在 run 目录再写一份权重。
            "state": file_hash(state_path) if kind == "source" else None,
        },
    }
    if variant is not None:
        report["variant"] = variant
    if mode in ("layers", "trace"):
        # 构造超参只作诊断证据（两端同名模块逐键比较），不进入判定
        write_json(out / "module_config.json", module_config(model))
    if module_shapes:
        write_json(out / "module_shapes.json", module_shapes)
    if mode == "layers":
        report["layers"] = len(layer_values)
    if kind == "source" and mode == "trace":
        report["module_probe"] = {
            "calls": len(module_calls),
            "input_tensors": len(module_inputs),
            "output_tensors": len(module_values),
            "calibration": calibration,
        }
    if isolated is not None:
        report["isolated"] = isolated
    if performance is not None:
        report["performance"] = performance
    write_json(out / "capture.json", report)
    return report


def main(job_path: str) -> int:
    from .training import ContractError

    job = read_json(Path(job_path))
    try:
        capture(job)
    except ContractError as exc:
        # 接口契约错误单独落盘：它是 Harness 读取任务契约时的失败，Agent 本地无法复现，
        # 由 run manifest 带回，供 diagnostic 返回冻结的枚举文本
        write_json(Path(job["out_dir"]) / "contract.json", exc.as_dict())
        raise
    return 0
