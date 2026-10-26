"""训练步等价：同一初始状态、同一冻结 batch 序列上执行 N 步更新。

记录每步 loss、步前学习率、可训练参数集合和每步更新后的可训练参数。完整的一步更新
（zero_grad、backward、裁剪、optimizer/scheduler step）由 task 的 `step` 入口完成，
TorchPort 只驱动步数并读取结果，不替仓库实现训练循环。
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import numpy as np

from .inputs import input_key, pack_values
from .model import _from_numpy, _lookup, _to_numpy


def enable_determinism(kind: str) -> None:
    """source oracle 必须逐位可重复；backward 默认的原子加做不到，必须在 CUDA 工作前开启。"""
    if kind != "source":
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _callable(scope: dict[str, Any], name: str) -> Any:
    value = _lookup(scope, name)
    if not callable(value):
        raise ValueError(f"入口中没有可调用的训练函数：{name}")
    return value


def _optimizer(state: Any) -> Any:
    # optimizer 工厂可返回 optimizer，或 (optimizer, scheduler, ...)；后者整体交给 step。
    return state[0] if isinstance(state, (tuple, list)) else state


def _lr(optimizer: Any) -> float:
    groups = getattr(optimizer, "param_groups", None) or []
    if groups and groups[0].get("lr") is not None:
        return float(groups[0]["lr"])
    lr = getattr(optimizer, "lr", None)
    if lr is None:
        raise ValueError("优化器没有可读取的学习率")
    return float(lr)


def _requires_grad(param: Any) -> bool:
    # PyTorch 用 requires_grad；Jittor 以 stop_grad 冻结参数。
    if hasattr(param, "is_stop_grad"):
        return not param.is_stop_grad()
    return bool(getattr(param, "requires_grad", True))


def _trainable(model: Any, optimizer: Any) -> dict[str, Any]:
    """可训练参数 = 优化器参数组中仍需梯度的模型参数；冻结与“不交给优化器”两种写法都覆盖。"""
    names = {id(param): name for name, param in model.named_parameters()}
    selected: dict[str, Any] = {}
    for group in getattr(optimizer, "param_groups", None) or []:
        for param in group.get("params", []):
            if id(param) not in names:
                raise ValueError("优化器包含模型之外的参数")
            if _requires_grad(param):
                selected[names[id(param)]] = param
    if not selected:
        raise ValueError("优化器中没有可训练参数")
    return dict(sorted(selected.items()))


def _batches(
    arrays: dict[str, np.ndarray],
    specs: list[dict[str, Any]],
    steps: int,
    kind: str,
    device: str,
    dtype: Any = None,
) -> Any:
    """输入的第 0 维是步号；第 t 步取各输入的第 t 个切片。"""
    for step in range(steps):
        values = []
        for index, spec in enumerate(specs):
            value = np.asarray(arrays[input_key(index, spec)])[step]
            if dtype is not None and np.issubdtype(value.dtype, np.floating):
                value = value.astype(dtype)
            tensor = _from_numpy(np.ascontiguousarray(value), kind)
            if kind == "source" and device == "cuda":
                tensor = tensor.cuda()
            values.append(tensor)
        yield pack_values(values, specs)


def _scalar(value: Any, kind: str) -> np.ndarray:
    array = _to_numpy(value, kind)
    if array.size != 1:
        raise ValueError(f"step 必须返回标量 loss，实际形状 {array.shape}")
    # 两框架的标量形状约定不同（() 与 [1]）；loss 统一为 [1]，dtype 仍按实际值比较。
    return array.reshape(1)


def run_steps(
    model: Any,
    kind: str,
    device: str,
    scope: dict[str, Any],
    train: dict[str, Any],
    arrays: dict[str, np.ndarray],
    specs: list[dict[str, Any]],
    steps: int,
    *,
    dtype: Any = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model.train()
    state = _callable(scope, train["optimizer"])(model)
    optimizer = _optimizer(state)
    trainable = _trainable(model, optimizer)
    requested = train.get("observe")
    if requested is not None:
        missing = sorted(set(requested) - set(trainable))
        if missing:
            raise ValueError(f"指定观察的参数不在可训练集合中：{missing}")
        observed = {name: trainable[name] for name in requested}
    else:
        observed = trainable
    step = _callable(scope, train["step"])
    outputs: dict[str, np.ndarray] = {}
    lrs: list[float] = []
    for index, (args, kwargs) in enumerate(_batches(arrays, specs, steps, kind, device, dtype)):
        lrs.append(_lr(optimizer))
        outputs[f"output.loss.{index}"] = _scalar(step(model, state, *args, **kwargs), kind)
        for name, param in observed.items():
            outputs[f"output.param.{index}.{name}"] = _to_numpy(param, kind)
    info = {
        "steps": steps,
        "trainable": list(trainable),
        "trainable_elements": int(sum(np.prod(tuple(p.shape)) for p in trainable.values())),
        "lr": lrs,
    }
    if requested is not None:
        info["observed"] = list(observed)
        info["observed_elements"] = int(sum(np.prod(tuple(p.shape)) for p in observed.values()))
    return outputs, info


def calibrate_steps(
    initial: Any,
    device: str,
    scope: dict[str, Any],
    train: dict[str, Any],
    arrays: dict[str, np.ndarray],
    specs: list[dict[str, Any]],
    steps: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """source 噪声底：同一初始状态分别跑 float64 真值和另一条 FP32 路径，覆盖完整 N 步。"""
    import torch

    def fresh() -> Any:
        model = deepcopy(initial)
        return model.cuda() if device == "cuda" else model

    torch.manual_seed(seed)
    fp64, _ = run_steps(
        fresh().double(), "source", device, scope, train, arrays, specs, steps, dtype=np.float64
    )
    backends = torch.backends
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        backends.cudnn.benchmark,
        backends.cudnn.allow_tf32,
        backends.cuda.matmul.allow_tf32,
    )
    try:
        # 另一条合法 FP32 路径：允许非确定性算法与 cuDNN 选算法，TF32 保持关闭。
        torch.use_deterministic_algorithms(False)
        backends.cudnn.benchmark = True
        backends.cudnn.allow_tf32 = backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(seed)
        alt, _ = run_steps(fresh(), "source", device, scope, train, arrays, specs, steps)
    finally:
        torch.use_deterministic_algorithms(saved[0])
        backends.cudnn.benchmark = saved[1]
        backends.cudnn.allow_tf32, backends.cuda.matmul.allow_tf32 = saved[2], saved[3]
    return fp64, alt
