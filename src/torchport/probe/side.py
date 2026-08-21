from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_data, make_data, to_values
from .model import add_hooks, func_scopes, load_model, load_scope, to_arrays
from .ops import OpTrace
from .paths import import_paths
from .state import load_checkpoint, load_state, save_state


def _digest(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values.values():
        data = np.ascontiguousarray(value)
        digest.update(str(data.dtype).encode())
        digest.update(str(data.shape).encode())
        digest.update(data.tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(kind: str, seed: int) -> dict[str, str]:
    if kind == "source":
        import torch

        torch.manual_seed(seed)
        return {"name": "pytorch", "version": torch.__version__}
    import jittor as jt

    jt.set_global_seed(seed)
    return {"name": "jittor", "version": jt.__version__}


def _check_bridge(kind: str) -> None:
    if kind != "target":
        return
    import jittor as jt

    expected = np.asarray([-1.25, 0.0, 2.5], dtype=np.float32)
    observed = np.asarray(jt.array(expected).numpy())
    if not np.array_equal(expected, observed):
        raise ValueError(
            f"Jittor/NumPy 数组桥接不兼容：NumPy {np.__version__}；请使用 NumPy < 2"
        )


def capture(
    kind: str,
    root: str | Path,
    *,
    entry: str,
    model_name: str,
    inputs: list[dict[str, Any]],
    model_args: list[Any],
    model_kw: dict[str, Any],
    seed: int,
    output: str | Path,
    state: str | Path,
    source: str | Path | None = None,
    weights: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output).resolve()
    state = Path(state).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    framework = _seed(kind, seed)
    _check_bridge(kind)
    original = list(sys.path)
    sys.path[:0] = [str(path) for path in import_paths(root)]
    try:
        scope = load_scope(root, entry)
        model = load_model(scope, model_name, kind, model_args, model_kw)
    finally:
        sys.path[:] = original
    model.eval()
    checkpoint = {"requested": False, "complete": True}

    if kind == "source":
        if weights is not None:
            weight_path = Path(weights).expanduser()
            if not weight_path.is_absolute():
                weight_path = root / weight_path
            weight_path = weight_path.resolve()
            checkpoint = load_checkpoint(model, weight_path)
            checkpoint["sha256"] = _file_hash(weight_path)
        arrays = make_data(inputs, seed)
        total = save_state(model, state)
        loaded = total
    elif source is not None:
        source_path = Path(source)
        arrays = load_data(source_path, inputs)
        replay = dict(np.load(source_path))
        total, loaded = load_state(model, state)
    else:
        raise ValueError("目标端采集需要源端输入证据")
    if kind == "source":
        replay = {}

    args, kwargs = to_values(arrays, inputs, kind)
    values = dict(arrays)
    handles = add_hooks(model, kind, values)
    trace = OpTrace(kind, values, [scope, *func_scopes(model, kind)], replay=replay)
    trace.start()
    try:
        result = model(*args, **kwargs)
    finally:
        trace.stop()
    outputs = to_arrays(result, kind, "output")
    if not outputs:
        raise ValueError("模型输出中没有张量")
    if "output" not in outputs:
        outputs = {"output": next(iter(outputs.values())), **outputs}
    values.update(outputs)
    for handle in handles:
        handle.remove()
    np.savez(output, **values)
    report = {
        "kind": kind,
        "entry": entry,
        "model": model_name,
        "inputs": inputs,
        "tensors": len(values),
        "weights": total,
        "loaded": loaded,
        "checkpoint": checkpoint,
        "framework": framework,
        "numpy": np.__version__,
        "hashes": {
            "input": _digest(arrays),
            "output": _digest(outputs),
            "archive": _file_hash(output),
            "weights": _file_hash(state),
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
