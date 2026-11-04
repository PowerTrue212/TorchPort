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


# 接口契约错误：Harness 读取训练契约时失败。文本是冻结枚举，只描述接口约定，
# 不含模块、算子、张量名或数值位置，因此 diagnostic 与 localized 都可返回。
CONTRACT_MESSAGES = {
    "TRAIN_FUNCTION_MISSING": "The training entry does not define the required callable `{name}`.",
    "OPTIMIZER_NO_PARAM_GROUPS": (
        "build_optimizer must return the optimizer, or a tuple whose first item is the "
        "optimizer; the optimizer must expose `param_groups`."
    ),
    "OPTIMIZER_NO_LR": "Each optimizer param group must have an `lr` entry for the next step.",
    "OPTIMIZER_FOREIGN_PARAM": "The optimizer holds parameters that are not model parameters.",
    "OPTIMIZER_NO_TRAINABLE": "The optimizer holds no trainable model parameters.",
    "LOSS_NOT_SCALAR": "train_step must return a loss with exactly one element.",
    "LOSS_TYPE": (
        "train_step must return the loss as a framework tensor, a NumPy scalar, "
        "or a Python number."
    ),
}


class ContractError(ValueError):
    def __init__(self, code: str, **fields: str) -> None:
        self.code = code
        self.message = CONTRACT_MESSAGES[code].format(**fields)
        super().__init__(f"{code}: {self.message}")

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def _callable(scope: dict[str, Any], name: str) -> Any:
    try:
        value = _lookup(scope, name)
    except Exception:  # noqa: BLE001 - 找不到即契约错误
        value = None
    if not callable(value):
        raise ContractError("TRAIN_FUNCTION_MISSING", name=name)
    return value


def _optimizer(state: Any) -> Any:
    # optimizer 工厂可返回 optimizer，或 (optimizer, scheduler, ...)；后者整体交给 step。
    return state[0] if isinstance(state, (tuple, list)) and state else state


def _lr(optimizer: Any) -> float:
    groups = getattr(optimizer, "param_groups", None) or []
    if groups and isinstance(groups[0], dict) and groups[0].get("lr") is not None:
        return float(groups[0]["lr"])
    lr = getattr(optimizer, "lr", None)
    if lr is None:
        raise ContractError("OPTIMIZER_NO_LR")
    return float(lr)


def _requires_grad(param: Any) -> bool:
    # PyTorch 用 requires_grad；Jittor 以 stop_grad 冻结参数。
    if hasattr(param, "is_stop_grad"):
        return not param.is_stop_grad()
    return bool(getattr(param, "requires_grad", True))


def _trainable(model: Any, optimizer: Any) -> dict[str, Any]:
    """可训练参数 = 优化器参数组中仍需梯度的模型参数；冻结与“不交给优化器”两种写法都覆盖。

    共享模块（如 RAFT 的 norm3 与 downsample.1）在 Jittor 的 named_parameters() 中按每个名字
    各出现一次，PyTorch 则去重并保留先注册的名字。这里同样取先出现的名字，两端规范名一致。
    """
    names: dict[int, str] = {}
    for name, param in model.named_parameters():
        names.setdefault(id(param), name)
    groups = getattr(optimizer, "param_groups", None)
    if not groups or not all(isinstance(group, dict) for group in groups):
        raise ContractError("OPTIMIZER_NO_PARAM_GROUPS")
    selected: dict[str, Any] = {}
    for group in groups:
        for param in group.get("params", []):
            if id(param) not in names:
                raise ContractError("OPTIMIZER_FOREIGN_PARAM")
            if _requires_grad(param):
                selected[names[id(param)]] = param
    if not selected:
        raise ContractError("OPTIMIZER_NO_TRAINABLE")
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


def _scalar(value: Any, kind: str, dtype: Any = None) -> np.ndarray:
    if isinstance(value, bool) or value is None:
        raise ContractError("LOSS_TYPE")
    if isinstance(value, (int, float, np.generic, np.ndarray)):
        # 非张量 loss（Jittor 没有 0 维 Var 时常见）：主路径 float32，fp64 参考路径 float64
        array = np.asarray(value, dtype=dtype or np.float32)
    elif hasattr(value, "numpy") or hasattr(value, "detach"):
        array = _to_numpy(value, kind)
    else:
        raise ContractError("LOSS_TYPE")
    if array.size != 1:
        raise ContractError("LOSS_NOT_SCALAR")
    # 两框架的标量形状约定不同（() 与 [1]）；loss 统一为 [1]，dtype 仍按实际值比较。
    return array.reshape(1)


_HASH_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def update_sketch(update: np.ndarray, dim: int) -> np.ndarray:
    """更新量的 CountSketch：元素数 ≤ dim 原样保存，否则按固定哈希与符号压到 dim 个桶。

    哈希只由 (元素数, dim) 决定，两端和三条参考路径用同一投影；投影线性，
    sketch(a) - sketch(b) = sketch(a - b)，任何非零偏差几乎必然落到某个桶上。
    """
    flat = np.asarray(update, dtype=np.float64).reshape(-1)
    if flat.size <= dim:
        return flat
    key = (flat.size, dim)
    if key not in _HASH_CACHE:
        rng = np.random.default_rng([flat.size, dim])
        _HASH_CACHE[key] = (
            rng.integers(0, dim, flat.size),
            rng.choice(np.asarray([-1.0, 1.0]), flat.size),
        )
    bucket, sign = _HASH_CACHE[key]
    return np.bincount(bucket, weights=sign * flat, minlength=dim)


def _update_stats(update: np.ndarray) -> np.ndarray:
    flat = np.asarray(update, dtype=np.float64).reshape(-1)
    if not flat.size:
        return np.zeros(3)
    return np.asarray([np.sqrt(np.square(flat).sum()), np.abs(flat).max(), flat.sum()])


def _grad_norm(trainable: dict[str, Any], kind: str) -> float | None:
    """source 侧全局梯度范数，只作覆盖事实（如裁剪是否触发），不参与比较。"""
    if kind != "source":
        return None
    total = 0.0
    found = False
    for param in trainable.values():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        found = True
        total += float(grad.detach().double().square().sum().cpu())
    return float(np.sqrt(total)) if found else None


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
    sketch_dim = train.get("sketch_dim")
    # sketch 模式比较相对初始状态的更新量；初值在 host 上以 fp64 保存，不依赖 candidate 代码
    initial = (
        {name: np.asarray(_to_numpy(param, kind), dtype=np.float64) for name, param in trainable.items()}
        if sketch_dim else {}
    )
    step = _callable(scope, train["step"])
    outputs: dict[str, np.ndarray] = {}
    lrs: list[float] = []
    grad_norms: list[float | None] = []
    for index, (args, kwargs) in enumerate(_batches(arrays, specs, steps, kind, device, dtype)):
        lrs.append(_lr(optimizer))
        outputs[f"output.loss.{index}"] = _scalar(
            step(model, state, *args, **kwargs), kind, dtype=dtype
        )
        if kind == "source":
            grad_norms.append(_grad_norm(trainable, kind))
        if sketch_dim:
            for name, param in trainable.items():
                update = np.asarray(_to_numpy(param, kind), dtype=np.float64) - initial[name]
                outputs[f"output.update.{index}.{name}"] = update_sketch(update, sketch_dim)
                outputs[f"output.ustat.{index}.{name}"] = _update_stats(update)
            continue
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
    if sketch_dim:
        info["sketch_dim"] = sketch_dim
    if kind == "source":
        # 覆盖事实：只记录，不参与比较；报告据此列出未执行的分支
        info["coverage"] = {
            "lr_changed": len(set(lrs)) > 1,
            "grad_norms": grad_norms,
        }
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
