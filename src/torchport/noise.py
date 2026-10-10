"""target 环境噪声下限 N_env：与候选无关，按算子在固定输入上重复运行，取运行间最大相对差。

协议 v3 的上限是 k * max(D, N_env)。N_env 只由 target 环境（解释器、环境变量、设备）和探针
规格决定，不读取候选代码，因此不稳定的候选不能借此放宽自己的上限。结果按环境身份缓存。
GPU 型号不进入身份（不运行就无法得知），证据目录本来按机器隔离，结果里另记 GPU 名称。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .execution import (
    empty_physical_cost,
    execute,
    new_run_dir,
    read_json,
    run_physical_cost,
    write_json,
)
from .task import CAPTURE_VERSION, digest

# 形状默认取 PlantCamo（704 输入）stage1：64x176x176 特征图、PVT block1 的 30976 token、sr=8 后 484 key。
# reduce_shape（[C, H, W]）只用于全局空间归约类算子；为 None 时沿用 channels/spatial。
DEFAULT_PROBE = {
    "repeats": 5, "seed": 0, "channels": 64, "spatial": 176, "tokens": 30976, "keys": 484,
    "reduce_shape": None,
}
# 在空间维上做统计的模块类（两端类名相同即可识别；函数式写法 x.mean((2,3)) 识别不到）。
REDUCE_CLASSES = {
    "AdaptiveAvgPool2d", "AdaptiveMaxPool2d", "GroupNorm", "InstanceNorm2d", "InstanceNorm1d",
}


# 模块类 -> 探针中对应的算子；未列出的类（Sequential、仓库自定义块等复合模块）用全局最大值。
OP_FAMILIES = {
    "Conv1d": ("conv_k3s1", "conv_k3s1_d6", "conv_k1", "conv_k8s8"),
    "Conv2d": ("conv_k3s1", "conv_k3s1_d6", "conv_k1", "conv_k8s8"),
    "ConvTranspose2d": ("conv_k3s1", "conv_k3s1_d6", "conv_k1", "conv_k8s8"),
    "Linear": ("linear",),
    "LayerNorm": ("layernorm",),
    "GroupNorm": ("group_norm",),
    "InstanceNorm1d": ("group_norm",),
    "InstanceNorm2d": ("group_norm",),
    "AdaptiveAvgPool2d": ("avgpool_global", "sum_spatial"),
    "AdaptiveMaxPool2d": ("avgpool_global", "sum_spatial"),
    "Softmax": ("softmax",),
    # eval 下逐元素：用实测的逐元素算子（gelu）代表
    "BatchNorm1d": ("gelu",), "BatchNorm2d": ("gelu",), "ReLU": ("gelu",), "GELU": ("gelu",),
    "Sigmoid": ("gelu",), "Tanh": ("gelu",), "SiLU": ("gelu",), "Dropout": ("gelu",),
    "Identity": ("gelu",), "UpsamplingBilinear2d": ("interp_up2",), "Upsample": ("interp_up2",),
}


def _module_of(name: str, config: dict[str, dict[str, Any]]) -> str | None:
    """张量名 `<module>`、`<module>#2`、`<module>.0` -> 模块名。"""
    base = name.split("#", 1)[0]
    if base in config:
        return base
    head, _, tail = base.rpartition(".")
    return head if head in config and tail.isdigit() else None


def per_tensor_floors(
    names: list[str], config: dict[str, dict[str, Any]], per_op: dict[str, float], default: float
) -> dict[str, float]:
    """按张量所属模块的类取对应算子的 N_env；模块类未知或探针缺该算子时用 default（全局最大值）。

    只用于 trace 隔离：每个模块的输入是 source 真实局部输入，相关噪声只有该算子自身的。
    """
    floors: dict[str, float] = {}
    for name in names:
        module = _module_of(name, config)
        ops = OP_FAMILIES.get((config.get(module) or {}).get("class", "")) if module else None
        values = [per_op[op] for op in ops or () if op in per_op]
        floors[name] = max(values) if ops and values else default
    return floors


def reduce_shape(config: dict[str, dict[str, Any]], shapes: dict[str, list[int]]) -> list[int] | None:
    """从 source 参考运行的模块类名与输入形状，取全局归约模块中空间规模最大的 [C, H, W]。

    AdaptiveXxxPool2d 只在输出为 1x1（全局池化）时计入；N_env 随归约元素数增长，取最大规模。
    """
    best: list[int] | None = None
    for name, entry in config.items():
        cls = entry.get("class")
        shape = shapes.get(name)
        if cls not in REDUCE_CLASSES or not shape or len(shape) != 4:
            continue
        if cls.startswith("Adaptive") and entry.get("output_size") not in (1, [1], [1, 1]):
            continue
        candidate = [int(shape[1]), int(shape[2]), int(shape[3])]
        if best is None or candidate[1] * candidate[2] > best[1] * best[2]:
            best = candidate
    return best


def _rel_spread(runs: list[np.ndarray]) -> float:
    """重复运行两两之间的最大相对 L2 距离（相对第一次运行的范数）。"""
    first = runs[0].astype(np.float64).reshape(-1)
    scale = max(float(np.sqrt(np.sum(first * first))), np.finfo(np.float64).tiny)
    worst = 0.0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            diff = (runs[i].astype(np.float64) - runs[j].astype(np.float64)).reshape(-1)
            worst = max(worst, float(np.sqrt(np.sum(diff * diff))) / scale)
    return worst


def summarize(runs: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    spread = {name: _rel_spread(values) for name, values in runs.items()}
    dominant = max(spread, key=spread.__getitem__) if spread else None
    return {
        "noise_floor": spread[dominant] if dominant else 0.0,
        "dominant_op": dominant,
        "per_op": spread,
    }


def run_probe(job: dict[str, Any]) -> dict[str, Any]:
    """在 target 子进程内运行：只导入 Jittor，不导入候选代码。"""
    import jittor as jt

    if job["side"].get("device", "cpu") == "cuda":
        jt.flags.use_cuda = 1
    probe = {**DEFAULT_PROBE, **(job.get("probe") or {})}
    rng = np.random.default_rng(int(probe["seed"]))
    c, s, n, m = (int(probe[key]) for key in ("channels", "spatial", "tokens", "keys"))

    def rand(*shape: int, scale: float = 1.0) -> np.ndarray:
        return (scale * rng.standard_normal(shape)).astype(np.float32)

    data = {
        "x_map": rand(1, c, s, s),
        "x_small": rand(1, c, s // 4, s // 4),
        "w_k3": rand(c, c, 3, 3, scale=0.05), "b_k3": rand(c, scale=0.1),
        "w_k1": rand(c // 4, c, 1, 1, scale=0.1), "b_k1": rand(c // 4, scale=0.1),
        "w_k8": rand(c, c, 8, 8, scale=0.02), "b_k8": rand(c, scale=0.1),
        "x_tok": rand(1, n, c),
        "w_lin": rand(4 * c, c, scale=0.1), "b_lin": rand(4 * c, scale=0.1),
        "q": rand(1, 1, n, c), "k": rand(1, 1, m, c), "v": rand(1, 1, m, c),
        "ln_w": rand(c, scale=0.5) + 1, "ln_b": rand(c, scale=0.1),
    }
    # 全局归约单独用任务真实规模的张量；卷积等算子仍用默认形状，避免大通道卷积占满显存
    rc, rh, rw = probe["reduce_shape"] or (c, s, s)
    data["x_reduce"] = rand(1, int(rc), int(rh), int(rw))
    g = {name: jt.array(value) for name, value in data.items()}
    nn = jt.nn
    group_norm = nn.GroupNorm(8 if int(rc) % 8 == 0 else 1, int(rc))
    group_norm.eval()

    def ops() -> dict[str, Any]:
        att = jt.matmul(g["q"], g["k"].transpose(0, 1, 3, 2)) * (c ** -0.5)
        weights = nn.softmax(att, dim=-1)
        return {
            "conv_k3s1": nn.conv2d(g["x_map"], g["w_k3"], g["b_k3"], padding=1),
            "conv_k3s1_d6": nn.conv2d(g["x_map"], g["w_k3"], g["b_k3"], padding=6, dilation=6),
            "conv_k1": nn.conv2d(g["x_map"], g["w_k1"], g["b_k1"]),
            "conv_k8s8": nn.conv2d(g["x_map"], g["w_k8"], g["b_k8"], stride=8),
            "avgpool_global": nn.AdaptiveAvgPool2d(1)(g["x_reduce"]),
            "sum_spatial": g["x_reduce"].sum(dims=(2, 3)),
            "mean_lastdim": g["x_tok"].mean(-1),
            "group_norm": group_norm(g["x_reduce"]),
            "linear": nn.matmul_transpose(g["x_tok"], g["w_lin"]) + g["b_lin"],
            "matmul_qk": att,
            "softmax": weights,
            "attn_av": jt.matmul(weights, g["v"]),
            "layernorm": nn.layer_norm(g["x_tok"], (c,), g["ln_w"], g["ln_b"], 1e-6),
            "gelu": nn.gelu(g["x_tok"]),
            "interp_up2": nn.interpolate(g["x_small"], scale_factor=2, mode="bilinear", align_corners=False),
        }

    runs: dict[str, list[np.ndarray]] = {}
    with jt.no_grad():
        for _ in range(int(probe["repeats"])):
            for name, value in ops().items():
                runs.setdefault(name, []).append(np.array(value.numpy(), copy=True))
            jt.sync_all()
            jt.gc()
    gpu = None
    if jt.flags.use_cuda:
        try:
            import subprocess

            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip().splitlines()[0]
        except Exception:  # 记录不到型号不影响标定本身
            gpu = None
    result = {
        "kind": "torchport.noise_probe",
        "probe": probe,
        "jittor": jt.__version__,
        "gpu": gpu,
        **summarize(runs),
    }
    write_json(Path(job["out_dir"]) / "noise.json", result)
    return result


def main(job_path: str) -> int:
    run_probe(read_json(Path(job_path)))
    return 0


def noise_identity(side: dict[str, Any], probe: dict[str, Any]) -> str:
    """只含 target 环境与探针规格；候选代码不参与。"""
    config = {key: side.get(key) for key in ("python", "env", "device")}
    return digest({**config, "probe": probe, "capture_version": CAPTURE_VERSION})


def probe_spec(task: dict[str, Any], envelope: dict[str, Any] | None = None) -> dict[str, Any]:
    """探针规格：默认值 < 由 source 参考运行推得的归约规模 < TaskSpec 显式 noise_probe。"""
    probe = dict(DEFAULT_PROBE)
    config_path = (envelope or {}).get("module_config")
    shapes_path = (envelope or {}).get("module_shapes")
    if config_path and shapes_path and Path(config_path).is_file() and Path(shapes_path).is_file():
        if shape := reduce_shape(read_json(Path(config_path)), read_json(Path(shapes_path))):
            probe["reduce_shape"] = shape
    probe.update(task["protocol"].get("noise_probe") or {})
    return probe


def calibrate(task: dict[str, Any], root: Path, envelope: dict[str, Any] | None = None) -> dict[str, Any]:
    """返回 target 环境的 N_env；同一环境与探针规格只实测一次。

    envelope 给定时，全局归约探针按 source 参考运行中最大的全局归约输入形状实测（形状只来自
    source，与候选无关）。
    """
    side = task["target"]
    probe = probe_spec(task, envelope)
    key = noise_identity(side, probe)
    path = Path(root) / "noise" / f"{key[:16]}.json"
    if path.is_file():
        meta = read_json(path)
        if meta.get("ok") and meta.get("identity") == key:
            return {**meta, "reused": True, "physical_cost": empty_physical_cost()}
    run_dir = new_run_dir(Path(root) / "runs", "noise", "target")
    write_json(run_dir / "job.json", {"side": side, "probe": probe, "out_dir": str(run_dir)})
    record = execute(
        side,
        ["_noise", str(run_dir / "job.json")],
        run_dir=run_dir,
        cache=Path(root) / "cache" / "noise" / key[:16],
        timeout=float(task["protocol"]["timeout"]),
    )
    physical = run_physical_cost({"execution": record}, side.get("device", "cpu"))
    out = run_dir / "noise.json"
    if not (record["ok"] and out.is_file()):
        return {
            "ok": False,
            "reason": "NOISE_PROBE_FAILED",
            "stderr_tail": record.get("stderr_tail"),
            "physical_cost": physical,
        }
    meta = {**read_json(out), "ok": True, "identity": key, "run_dir": str(run_dir)}
    write_json(path, meta)
    return {**meta, "reused": False, "physical_cost": physical}
