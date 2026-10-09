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
DEFAULT_PROBE = {"repeats": 5, "seed": 0, "channels": 64, "spatial": 176, "tokens": 30976, "keys": 484}


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
    g = {name: jt.array(value) for name, value in data.items()}
    nn = jt.nn
    group_norm = nn.GroupNorm(8, c)
    group_norm.eval()

    def ops() -> dict[str, Any]:
        att = jt.matmul(g["q"], g["k"].transpose(0, 1, 3, 2)) * (c ** -0.5)
        weights = nn.softmax(att, dim=-1)
        return {
            "conv_k3s1": nn.conv2d(g["x_map"], g["w_k3"], g["b_k3"], padding=1),
            "conv_k3s1_d6": nn.conv2d(g["x_map"], g["w_k3"], g["b_k3"], padding=6, dilation=6),
            "conv_k1": nn.conv2d(g["x_map"], g["w_k1"], g["b_k1"]),
            "conv_k8s8": nn.conv2d(g["x_map"], g["w_k8"], g["b_k8"], stride=8),
            "avgpool_global": nn.AdaptiveAvgPool2d(1)(g["x_map"]),
            "sum_spatial": g["x_map"].sum(dims=(2, 3)),
            "mean_lastdim": g["x_tok"].mean(-1),
            "group_norm": group_norm(g["x_map"]),
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


def calibrate(task: dict[str, Any], root: Path) -> dict[str, Any]:
    """返回 target 环境的 N_env；同一环境与探针规格只实测一次。"""
    side = task["target"]
    probe = {**DEFAULT_PROBE, **(task["protocol"].get("noise_probe") or {})}
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
