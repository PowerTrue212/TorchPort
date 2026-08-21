from __future__ import annotations

from typing import Any

from .data import resolve_inputs


def merge_probe(
    base: dict[str, Any],
    *,
    entry: str | None = None,
    model: str | None = None,
    inputs: list[dict[str, Any]] | None = None,
    shapes: tuple[tuple[int, ...], ...] | None = None,
    dtypes: tuple[str, ...] | None = None,
    model_args: list[Any] | None = None,
    model_kw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(base)
    if entry is not None:
        result["entry"] = entry
    if model is not None:
        result["model"] = model
        _select(result, model)
    if model_args is not None:
        result["args"] = model_args
    else:
        result.setdefault("args", [])
    if model_kw is not None:
        result["kwargs"] = model_kw
    else:
        result.setdefault("kwargs", {})
    if model_args is not None or model_kw is not None:
        result["required"] = []
    else:
        result.setdefault("required", [])
    result.setdefault("env", {})

    resolved = resolve_inputs(result, inputs, shapes, dtypes)
    result["inputs"] = resolved
    if len(resolved) == 1:
        result["shape"] = resolved[0]["shape"]
        result["dtype"] = resolved[0].get("dtype", "float32")
    else:
        result["shape"] = None
        types = {item.get("dtype", "float32") for item in resolved}
        result["dtype"] = next(iter(types)) if len(types) == 1 else None

    if not result.get("entry"):
        result["status"] = "missing"
        result["reason"] = "没有运行入口"
    elif not result.get("model"):
        result["status"] = "missing"
        result["reason"] = "仓库中没有可访问的模型或工厂"
    elif result.get("required"):
        result["status"] = "needs-args"
        result.pop("reason", None)
    elif not result["inputs"]:
        result["status"] = "needs-input"
        result.pop("reason", None)
    else:
        result["status"] = "ready"
        result.pop("reason", None)
    return result


def _select(probe: dict[str, Any], model: str) -> None:
    for item in probe.get("candidates", []):
        name = str(item.get("name") or "")
        if name == model or name.endswith(f".{model}") or name == model.replace(":", "."):
            probe["required"] = list(item.get("required") or [])
            break
