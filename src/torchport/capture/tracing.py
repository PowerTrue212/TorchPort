"""算子与模块 trace、source 标定和 target 模块隔离。"""

from __future__ import annotations

import inspect
import json
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .model import (
    _decode_call_value,
    _encode_call_value,
    _fw,
    _synchronize,
    find_tensor,
    to_array,
    to_arrays,
)

OP_NAMES = (
    "relu",
    "gelu",
    "silu",
    "softmax",
    "log_softmax",
    "max_pool2d",
    "avg_pool2d",
    "adaptive_avg_pool2d",
    "interpolate",
    "normalize",
    "pad",
)
RAND_NAMES = ("rand", "randn")


def _library_call() -> bool:
    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame and frame.f_back else None
    filename = caller.f_code.co_filename.replace("\\", "/") if caller else ""
    del frame
    return "/site-packages/torch/" in filename or "/site-packages/jittor/" in filename


class OpTrace:
    """临时包装常见函数式算子并记录真实输出。"""

    def __init__(
        self,
        kind: str,
        values: dict[str, np.ndarray],
        scopes: list[dict[str, Any]] | None = None,
    ) -> None:
        self.kind, self.values = kind, values
        self.scopes = scopes or []
        self.counts: Counter[str] = Counter()
        self.saved: list[tuple[Any, str, Callable]] = []
        self.locals: list[tuple[dict, str, Callable]] = []
        self.modules = self._modules(kind)

    @staticmethod
    def _modules(kind: str) -> list[tuple[Any, tuple[str, ...]]]:
        fw = _fw(kind)
        ops = fw.nn.functional if kind == "source" else fw.nn
        return [(ops, OP_NAMES), (fw, RAND_NAMES)]

    def start(self) -> None:
        for module, names in self.modules:
            for name in names:
                func = getattr(module, name, None)
                if not callable(func):
                    continue
                self.saved.append((module, name, func))
                wrapped = self._wrap(name, func)
                setattr(module, name, wrapped)
                for scope in self.scopes:
                    for local, value in list(scope.items()):
                        if value is func:
                            self.locals.append((scope, local, func))
                            scope[local] = wrapped

    def stop(self) -> None:
        for module, name, func in self.saved:
            setattr(module, name, func)
        for scope, name, func in self.locals:
            scope[name] = func
        self.saved, self.locals = [], []

    def _wrap(self, name: str, func: Callable) -> Callable:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _library_call():
                return func(*args, **kwargs)
            self.counts[name] += 1
            count = self.counts[name]
            key = f"op.{name}" if count == 1 else f"op.{name}#{count}"
            output = func(*args, **kwargs)
            tensor = find_tensor(output, self.kind)
            if tensor is not None:
                self.values[key] = to_array(tensor, self.kind)
            return output

        return wrapped


def _defined_in_repo(module: Any, root: Path) -> bool:
    try:
        path = Path(inspect.getsourcefile(type(module)) or "").resolve()
        path.relative_to(root.resolve())
        return True
    except (OSError, TypeError, ValueError):
        return False


def _has_local_state(module: Any) -> bool:
    """模块自身（不含子模块）是否持有参数或 buffer。

    不能用 state_dict(recurse=False)：PyTorch 的 state_dict 不接受 recurse，会抛 TypeError，
    结果所有内置层（LayerNorm、Conv2d、Linear）都不被采集局部输入，隔离比较漏掉 eps 这类错误。
    """
    try:
        return any(True for _ in module.parameters(recurse=False)) or any(
            True for _ in module.buffers(recurse=False)
        )
    except (AttributeError, TypeError):
        # 局部输入只在 source（PyTorch）端采集；其他框架没有 recurse 参数时视为无局部状态。
        return False


def add_hooks(
    model: Any,
    kind: str,
    values: dict[str, np.ndarray],
    *,
    root: Path | None = None,
    probe_values: dict[str, np.ndarray] | None = None,
    probe_calls: list[dict[str, Any]] | None = None,
    probe_inputs: dict[str, np.ndarray] | None = None,
    focus: set[str] | None = None,
    shapes: dict[str, list[int]] | None = None,
) -> list[Any]:
    """focus 给定时只记录这些模块的输出与局部输入（trace 聚焦首次分歧附近，避免 GB 级产物）。

    shapes 给定时记录每个模块首次调用的第一个输入张量形状（N_env 按真实归约规模标定用）。
    """
    counts: Counter[str] = Counter()
    handles = []

    def make(name: str, selected: bool) -> Callable:
        def record(_module: Any, inputs: Any, kwargs: dict[str, Any], output: Any) -> None:
            counts[name] += 1
            if shapes is not None and name not in shapes:
                tensor = find_tensor(inputs, kind)
                if tensor is not None:
                    shapes[name] = [int(x) for x in tensor.shape]
            if focus is not None and name not in focus:
                return
            key = name if counts[name] == 1 else f"{name}#{counts[name]}"
            output_arrays = to_arrays(output, kind, key)
            values.update(output_arrays)
            if not selected or probe_values is None or probe_calls is None or probe_inputs is None:
                return
            try:
                # 只在一次调用内部去重。跨调用缓存对象 id 会遇到 Python 复用已释放 id，
                # 从而把后续模块错误地指向较早的张量。
                tensor_inputs: dict[int, str] = {}
                call = {
                    "key": key,
                    "module": name,
                    "args": _encode_call_value(inputs, kind, probe_inputs, tensor_inputs),
                    "kwargs": _encode_call_value(kwargs, kind, probe_inputs, tensor_inputs),
                    "output_dtypes": {
                        output_name: str(value.dtype)
                        for output_name, value in output_arrays.items()
                    },
                }
            except TypeError as exc:
                probe_calls.append({"key": key, "module": name, "unsupported": str(exc)})
                return
            probe_calls.append(call)
            probe_values.update(output_arrays)

        if kind == "source":
            def hook(_module: Any, inputs: Any, kwargs: dict[str, Any], output: Any) -> None:
                record(_module, inputs, kwargs, output)
        else:
            def hook(_module: Any, inputs: Any, output: Any, *_extra: Any) -> None:
                record(_module, inputs, {}, output)

        return hook

    for name, module in model.named_modules():
        if not name:
            continue
        selected = kind == "source" and root is not None and (
            _defined_in_repo(module, root) or _has_local_state(module)
        ) and (focus is None or name in focus)
        hook = make(name, selected)
        if kind == "source":
            handle = module.register_forward_hook(hook, with_kwargs=True)
        else:
            handle = module.register_forward_hook(hook)
        if handle is not None:
            handles.append(handle)
    return handles


def _mark_leaves(calls: list[dict[str, Any]]) -> None:
    """叶子 = 没有任何其他被采集模块位于其名字之下；只有叶子参与噪声标定判定。"""
    names = {call["module"] for call in calls}
    for call in calls:
        prefix = call["module"] + "."
        call["leaf"] = not any(other.startswith(prefix) for other in names)


def _to_float64(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.double() if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(_to_float64(item) for item in value)
    if isinstance(value, list):
        return [_to_float64(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_float64(item) for key, item in value.items()}
    return value


def _float32_like_reference(
    values: dict[str, np.ndarray],
    dtypes: dict[str, str] | None,
) -> dict[str, np.ndarray]:
    """参考输出是 float32 时按 float32 落盘，避免后端无意中扩成 float64。"""
    dtypes = dtypes or {}
    return {
        name: (
            np.asarray(value, dtype=np.float32)
            if dtypes.get(name) == "float32" and np.issubdtype(np.asarray(value).dtype, np.floating)
            else np.asarray(value)
        )
        for name, value in values.items()
    }


def calibrate_source_modules(
    model: Any,
    calls: list[dict[str, Any]],
    inputs: dict[str, np.ndarray],
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """source 端噪声底：同一份局部输入分别跑 float64 真值和另一条 FP32 执行路径。"""
    import torch

    modules = dict(model.named_modules())
    fp64: dict[str, np.ndarray] = {}
    alt: dict[str, np.ndarray] = {}
    repeat: dict[str, np.ndarray] = {}
    failed: list[dict[str, str]] = []

    def decode(call: dict[str, Any]) -> tuple[Any, Any]:
        args = _decode_call_value(call["args"], inputs, "source")
        kwargs = _decode_call_value(call["kwargs"], inputs, "source")
        if device == "cuda":
            args = _move(args, "cuda")
            kwargs = _move(kwargs, "cuda")
        return args, kwargs

    backends = torch.backends
    saved = (
        backends.cudnn.benchmark,
        backends.cudnn.deterministic,
        backends.cudnn.allow_tf32,
        backends.cuda.matmul.allow_tf32,
    )
    with torch.no_grad():
        for call in calls:
            if call.get("unsupported") or call["module"] not in modules:
                continue
            key, module = call["key"], modules[call["module"]]
            try:
                args, kwargs = decode(call)
                # 另一条 FP32 路径：换 cuDNN 算法选择，TF32 保持关闭，只量化“合法的 FP32 差异”。
                backends.cudnn.benchmark, backends.cudnn.deterministic = True, False
                backends.cudnn.allow_tf32 = backends.cuda.matmul.allow_tf32 = False
                alt.update(
                    _float32_like_reference(
                        to_arrays(module(*args, **kwargs), "source", key),
                        call.get("output_dtypes"),
                    )
                )
                (
                    backends.cudnn.benchmark,
                    backends.cudnn.deterministic,
                    backends.cudnn.allow_tf32,
                    backends.cuda.matmul.allow_tf32,
                ) = saved
                # source 重复性：默认设置下同输入再跑一次，与 target 的 isolated_repeat 对称。
                repeat.update(
                    _float32_like_reference(
                        to_arrays(module(*args, **kwargs), "source", key),
                        call.get("output_dtypes"),
                    )
                )
                double = deepcopy(module).double()
                # fp64 是 v3 在约 1e-7 量级估计 source 路径距离的基准；降为
                # float32 会把这部分差异舍入掉，因此只有该归档保留 float64。
                fp64.update(to_arrays(double(*_to_float64(args), **_to_float64(kwargs)), "source", key))
                del double
            except Exception as exc:  # noqa: BLE001 - 标定失败只让该模块失去判定依据
                failed.append({"key": key, "reason": f"{type(exc).__name__}: {exc}"[:500]})
            finally:
                (
                    backends.cudnn.benchmark,
                    backends.cudnn.deterministic,
                    backends.cudnn.allow_tf32,
                    backends.cuda.matmul.allow_tf32,
                ) = saved
    if device == "cuda":
        torch.cuda.synchronize()
    return fp64, alt, repeat, {
        "fp64_tensors": len(fp64),
        "alt_tensors": len(alt),
        "repeat_tensors": len(repeat),
        "failed": failed,
    }


def _move(value: Any, device: str) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def run_isolated_modules(
    model: Any,
    oracle: Path,
    kind: str,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """用 source 的真实局部输入逐个运行同名 target 模块。"""
    calls_path = oracle / "module_calls.json"
    inputs_path = oracle / "module_inputs.npz"
    if not calls_path.is_file() or not inputs_path.is_file():
        return {}, {}, {"available": False, "reason": "source 没有模块输入证据"}
    calls = json.loads(calls_path.read_text("utf-8"))
    modules = dict(model.named_modules())
    values: dict[str, np.ndarray] = {}
    repeats: dict[str, np.ndarray] = {}
    missing: list[str] = []
    unsupported: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    succeeded = 0
    with np.load(inputs_path, allow_pickle=False) as arrays:
        for call in calls:
            key, name = call["key"], call["module"]
            if reason := call.get("unsupported"):
                unsupported.append({"key": key, "reason": str(reason)})
                continue
            module = modules.get(name)
            if module is None:
                missing.append(key)
                continue
            try:
                args = _decode_call_value(call["args"], arrays, kind)
                kwargs = _decode_call_value(call["kwargs"], arrays, kind)
                output = module(*args, **kwargs)
                _synchronize(kind, device, output)
                values.update(
                    _float32_like_reference(
                        to_arrays(output, kind, key),
                        call.get("output_dtypes"),
                    )
                )
                # 使用重新解码的同一份 source 输入再跑一次，单独测 target 自身波动。
                repeat_args = _decode_call_value(call["args"], arrays, kind)
                repeat_kwargs = _decode_call_value(call["kwargs"], arrays, kind)
                repeat = module(*repeat_args, **repeat_kwargs)
                _synchronize(kind, device, repeat)
                repeats.update(
                    _float32_like_reference(
                        to_arrays(repeat, kind, key),
                        call.get("output_dtypes"),
                    )
                )
                succeeded += 1
            except Exception as exc:  # noqa: BLE001 - 每个模块单独留下可审计失败
                failed.append({"key": key, "reason": f"{type(exc).__name__}: {exc}"[:500]})
    return values, repeats, {
        "available": True,
        "recorded": len(calls),
        "succeeded": succeeded,
        "missing_modules": missing,
        "unsupported_inputs": unsupported,
        "failed_modules": failed,
    }


def func_scopes(model: Any, kind: str) -> list[dict[str, Any]]:
    name = "forward" if kind == "source" else "execute"
    scopes, seen = [], set()
    for _path, module in model.named_modules():
        method = getattr(module, name, None)
        scope = getattr(getattr(method, "__func__", method), "__globals__", None)
        if isinstance(scope, dict) and id(scope) not in seen:
            seen.add(id(scope))
            scopes.append(scope)
    return scopes

