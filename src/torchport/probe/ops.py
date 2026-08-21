from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from .model import find_tensor, to_array

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


class OpTrace:
    """临时包装常见函数式算子并记录输出，结束后恢复原函数。"""

    def __init__(
        self,
        kind: str,
        values: dict[str, np.ndarray],
        scopes: list[dict[str, Any]] | None = None,
        replay: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.kind = kind
        self.values = values
        self.scopes = scopes or []
        self.replay = replay or {}
        self.counts: Counter[str] = Counter()
        self.saved: list[tuple[Any, str, Callable[..., Any]]] = []
        self.locals: list[tuple[dict[str, Any], str, Callable[..., Any]]] = []
        self.modules = self._modules(kind)

    @staticmethod
    def _modules(kind: str) -> list[tuple[Any, tuple[str, ...]]]:
        if kind == "source":
            import torch
            import torch.nn.functional as ops

            return [(ops, OP_NAMES), (torch, RAND_NAMES)]
        import jittor as jt
        import jittor.nn as ops

        return [(ops, OP_NAMES), (jt, RAND_NAMES)]

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
        self.saved = []
        self.locals.clear()

    def _wrap(self, name: str, func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _library_call():
                return func(*args, **kwargs)
            self.counts[name] += 1
            count = self.counts[name]
            key = f"op.{name}" if count == 1 else f"op.{name}#{count}"
            if self.kind == "target" and key in self.replay:
                output = self._tensor(self.replay[key])
            else:
                output = func(*args, **kwargs)
            tensor = find_tensor(output, self.kind)
            if tensor is not None:
                self.values[key] = to_array(tensor, self.kind)
            return output

        return wrapped

    def _tensor(self, value: np.ndarray) -> Any:
        if self.kind == "source":
            import torch

            return torch.from_numpy(np.asarray(value))
        import jittor as jt

        return jt.array(np.asarray(value))


def _library_call() -> bool:
    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame and frame.f_back else None
    filename = caller.f_code.co_filename.replace("\\", "/") if caller else ""
    del frame
    return "/site-packages/torch/" in filename or "/site-packages/jittor/" in filename
