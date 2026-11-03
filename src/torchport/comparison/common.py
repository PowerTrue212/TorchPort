"""数值比较共享的距离、状态与张量名称工具。"""

from __future__ import annotations

import numpy as np

SHAPE_PREFIX = "__shape__/"


def _l2(x: np.ndarray) -> float:
    # 不走 np.linalg.norm：它调用多线程 BLAS dot，在 65536 维 sketch 上线程调度开销
    # 比计算本身大约 4500 倍（48 线程实测 7e-2 s 对 1.6e-5 s）。ufunc 求和不进 BLAS。
    flat = x.astype(np.complex128 if np.iscomplexobj(x) else np.float64, copy=False).reshape(-1)
    return float(np.sqrt(np.sum(np.abs(flat) ** 2)))


def _rel(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    dtype = np.complex128 if np.iscomplexobj(a) or np.iscomplexobj(b) else np.float64
    return _l2(a.astype(dtype) - b.astype(dtype)) / scale


def _diameter(arrays: list[np.ndarray], scale: float) -> float:
    """一组合法运行两两之间的最大相对距离；少于两份时为 0。"""
    return max(
        (_rel(arrays[i], arrays[j], scale) for i in range(len(arrays)) for j in range(i + 1, len(arrays))),
        default=0.0,
    )


def _is_output(name: str) -> bool:
    return name == "output" or name.startswith("output.")


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    dtype = np.complex128 if np.iscomplexobj(a) or np.iscomplexobj(b) else np.float64
    return float(np.max(np.abs(a.astype(dtype) - b.astype(dtype)), initial=0.0))


def _band(ratio: float, pass_k: float, floor_k: float) -> str:
    return "pass" if ratio <= pass_k else ("marginal" if ratio <= floor_k else "fail")


def _leaf_key(name: str, leaves: set[str]) -> bool:
    """张量名形如 `<module>`、`<module>#2`、`<module>.0`；去掉调用序号与输出下标后匹配模块名。"""
    base = name.split("#", 1)[0]
    while base:
        if base in leaves:
            return True
        if "." not in base:
            return False
        head, tail = base.rsplit(".", 1)
        if not tail.isdigit():
            return False
        base = head
    return False

