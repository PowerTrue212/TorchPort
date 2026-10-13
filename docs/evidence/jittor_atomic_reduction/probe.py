"""Isolate Jittor CUDA reduction repeatability and the effect of accumulation order.

The probe keeps the input fixed and compares five implementations of the same
spatial sum.  ``explicit_atomic`` uses many CUDA blocks that merge partial
results with ``atomicAdd``.  ``fixed_tree`` assigns one block to each output and
uses a fixed shared-memory reduction tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Callable

import jittor as jt
import numpy as np
from jittor import nn

REPEATS = 30
SEED = 20261011
SHAPE = (1, 64, 176, 176)
THREADS = 256
BLOCKS_PER_OUTPUT = 8


ATOMIC_CUDA = rf"""
__global__ static void zero_output(@ARGS_DEF) {{
    @PRECALC
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_shape0 * out_shape1) out_p[i] = 0.0f;
}}

__global__ static void atomic_spatial_sum(@ARGS_DEF) {{
    @PRECALC
    const int output_count = out_shape0 * out_shape1;
    const int output_id = blockIdx.x / {BLOCKS_PER_OUTPUT};
    const int part = blockIdx.x % {BLOCKS_PER_OUTPUT};
    if (output_id >= output_count) return;
    const int n = output_id / in0_shape1;
    const int c = output_id % in0_shape1;
    const int spatial = in0_shape2 * in0_shape3;
    float local = 0.0f;
    for (int k = part * blockDim.x + threadIdx.x;
         k < spatial;
         k += blockDim.x * {BLOCKS_PER_OUTPUT}) {{
        local += @in0(n, c, k / in0_shape3, k % in0_shape3);
    }}
    atomicAdd(out_p + output_id, local);
}}

zero_output<<<(out_shape0 * out_shape1 + {THREADS} - 1) / {THREADS}, {THREADS}>>>(@ARGS);
atomic_spatial_sum<<<out_shape0 * out_shape1 * {BLOCKS_PER_OUTPUT}, {THREADS}>>>(@ARGS);
"""


FIXED_TREE_CUDA = rf"""
__global__ static void fixed_tree_spatial_sum(@ARGS_DEF) {{
    @PRECALC
    const int output_id = blockIdx.x;
    const int output_count = out_shape0 * out_shape1;
    if (output_id >= output_count) return;
    const int n = output_id / in0_shape1;
    const int c = output_id % in0_shape1;
    const int spatial = in0_shape2 * in0_shape3;
    __shared__ float partial[{THREADS}];
    float local = 0.0f;
    for (int k = threadIdx.x; k < spatial; k += blockDim.x) {{
        local += @in0(n, c, k / in0_shape3, k % in0_shape3);
    }}
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {{
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }}
    if (threadIdx.x == 0) out_p[output_id] = partial[0];
}}

fixed_tree_spatial_sum<<<out_shape0 * out_shape1, {THREADS}>>>(@ARGS);
"""


def explicit_atomic(x: jt.Var) -> jt.Var:
    return jt.code((x.shape[0], x.shape[1]), x.dtype, [x], cuda_src=ATOMIC_CUDA)


def fixed_tree(x: jt.Var) -> jt.Var:
    return jt.code((x.shape[0], x.shape[1]), x.dtype, [x], cuda_src=FIXED_TREE_CUDA)


def pairwise_metrics(runs: list[np.ndarray], exact: np.ndarray | None = None) -> dict[str, object]:
    flat = [np.asarray(value, dtype=np.float32).reshape(-1) for value in runs]
    scale_source = exact if exact is not None else flat[0].astype(np.float64)
    scale = max(float(np.linalg.norm(np.asarray(scale_source, dtype=np.float64).reshape(-1))), np.finfo(np.float64).tiny)
    max_abs = 0.0
    max_rel_l2 = 0.0
    max_mismatch = 0
    for i, left in enumerate(flat):
        for right in flat[i + 1 :]:
            delta = left.astype(np.float64) - right.astype(np.float64)
            max_abs = max(max_abs, float(np.max(np.abs(delta))))
            max_rel_l2 = max(max_rel_l2, float(np.linalg.norm(delta) / scale))
            max_mismatch = max(max_mismatch, int(np.count_nonzero(left.view(np.uint32) != right.view(np.uint32))))
    result: dict[str, object] = {
        "bitwise_equal": max_mismatch == 0,
        "max_mismatched_elements": max_mismatch,
        "max_abs_run_diff": max_abs,
        "max_relative_l2_run_diff": max_rel_l2,
    }
    if exact is not None:
        exact_flat = np.asarray(exact, dtype=np.float64).reshape(-1)
        result["max_relative_l2_error_vs_fp64"] = max(
            float(np.linalg.norm(value.astype(np.float64) - exact_flat) / scale) for value in flat
        )
    return result


def gpu_info() -> str | None:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip().splitlines()[0]
    except Exception:
        return None


def run_many(operations: dict[str, Callable[[], jt.Var]]) -> dict[str, list[np.ndarray]]:
    outputs = {name: [] for name in operations}
    with jt.no_grad():
        for _ in range(REPEATS):
            for name, operation in operations.items():
                value = operation()
                value.sync()
                outputs[name].append(np.array(value.numpy(), copy=True))
            jt.sync_all()
            jt.gc()
    return outputs


def main() -> None:
    jt.flags.use_cuda = 1
    rng = np.random.default_rng(SEED)
    x_np = rng.standard_normal(SHAPE).astype(np.float32)
    x = jt.array(x_np)
    weights = jt.array(rng.standard_normal((64, 64, 3, 3)).astype(np.float32) * 0.03)
    bias = jt.array(rng.standard_normal((64,)).astype(np.float32) * 0.1)

    operations = {
        "builtin_sum_spatial": lambda: x.sum(dims=(2, 3)),
        "builtin_avgpool": lambda: nn.AdaptiveAvgPool2d(1)(x),
        "explicit_atomic": lambda: explicit_atomic(x),
        "fixed_tree": lambda: fixed_tree(x),
        "two_stage_sum": lambda: x.sum(dim=3).sum(dim=2),
        "conv3x3": lambda: nn.conv2d(x, weights, bias, padding=1),
    }

    # Compile and warm every path before the measured repetitions.
    for operation in operations.values():
        operation().sync()
    jt.sync_all()

    outputs = run_many(operations)
    exact_sum = x_np.astype(np.float64).sum(axis=(2, 3))
    exact_avg = x_np.astype(np.float64).mean(axis=(2, 3), keepdims=True)
    exact_by_name = {
        "builtin_sum_spatial": exact_sum,
        "builtin_avgpool": exact_avg,
        "explicit_atomic": exact_sum,
        "fixed_tree": exact_sum,
        "two_stage_sum": exact_sum,
    }
    metrics = {
        name: pairwise_metrics(values, exact_by_name.get(name))
        for name, values in outputs.items()
    }
    result = {
        "schema_version": 1,
        "experiment": "jittor_atomic_reduction_causal_probe",
        "timestamp_note": "Run date is encoded by the containing evidence directory.",
        "environment": {
            "jittor": jt.__version__,
            "python": platform.python_version(),
            "gpu": gpu_info(),
            "conv_opt": os.environ.get("conv_opt"),
            "use_cutlass": os.environ.get("use_cutlass"),
            "jittor_home": os.environ.get("JITTOR_HOME"),
        },
        "protocol": {
            "seed": SEED,
            "shape": SHAPE,
            "dtype": "float32",
            "repeats": REPEATS,
            "threads": THREADS,
            "atomic_blocks_per_output": BLOCKS_PER_OUTPUT,
            "input_sha256": hashlib.sha256(x_np.tobytes()).hexdigest(),
        },
        "metrics": metrics,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out_dir = Path(os.environ.get("PROBE_OUT", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "outputs.npz",
        **{
            f"{name}/run_{i:02d}": value
            for name, values in outputs.items()
            if name != "conv3x3"
            for i, value in enumerate(values)
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
