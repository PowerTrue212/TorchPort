from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from common import ROOT, load_config, method_ids, read_json_lines, verify_manifest, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one migrated Jittor RAFT submission")
    parser.add_argument("--method", required=True, choices=method_ids())
    return parser.parse_args()


def import_submission(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("raft_migration_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import submission: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("load_model", "forward_flow"):
        if not callable(getattr(module, name, None)):
            raise TypeError(f"submission must define callable {name}")
    return module


def base_result(method: str, rounds: int, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": config["protocol_version"],
        "method": method,
        "rn": False,
        "na": False,
        "max_error": None,
        "mean_error": None,
        "mean_framework_flow_difference": None,
        "max_framework_flow_difference": None,
        "p99_framework_flow_difference": None,
        "rounds": rounds,
        "rtol": config["alignment"]["rtol"],
        "atol": config["alignment"]["atol"],
        "shape": None,
        "failure": None,
        "elapsed_seconds": None,
    }


def main() -> int:
    args = parse_args()
    config = load_config()
    verify_manifest("reference")
    result_dir = ROOT / "results" / args.method
    rounds = read_json_lines(result_dir / "rounds.jsonl")
    maximum = int(config["execution"]["max_repair_rounds"])
    if len(rounds) > maximum:
        raise RuntimeError(f"repair log exceeds frozen maximum {maximum}")

    result = base_result(args.method, len(rounds), config)
    attempt_dir = result_dir / "attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    attempt_number = len(rounds)
    started = time.perf_counter()
    flow_up_array: np.ndarray | None = None

    try:
        import jittor as jt

        if jt.__version__ != config["environment"]["jittor"]:
            raise RuntimeError(
                f"frozen protocol requires Jittor {config['environment']['jittor']}, "
                f"got {jt.__version__}"
            )
        if np.__version__ != config["environment"]["numpy"]:
            raise RuntimeError(
                f"frozen protocol requires NumPy {config['environment']['numpy']}, got {np.__version__}"
            )
        if not bool(jt.has_cuda):
            raise RuntimeError("Jittor CUDA is unavailable")
        gpu_names = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()
        if not any("V100" in name.upper() for name in gpu_names):
            raise RuntimeError(f"frozen protocol requires an NVIDIA V100, got {gpu_names}")
        result["gpu_names"] = gpu_names
        result["jittor_version"] = jt.__version__
        result["numpy_version"] = np.__version__
        jt.flags.use_cuda = 1

        prepared = ROOT / "shared" / "prepared"
        image1 = np.load(prepared / "image1.npy", allow_pickle=False)
        image2 = np.load(prepared / "image2.npy", allow_pickle=False)
        reference = np.load(prepared / "flow_up_ref.npy", allow_pickle=False)
        submission_path = ROOT / "workspaces" / args.method / "repo" / "submission.py"
        if not submission_path.is_file():
            raise FileNotFoundError(submission_path)

        torch_was_loaded = "torch" in sys.modules
        submission = import_submission(submission_path)
        if not torch_was_loaded and "torch" in sys.modules:
            raise RuntimeError("submission imported PyTorch in the Jittor evaluator")
        model = submission.load_model(prepared / "weights_fp32.npz")
        if not isinstance(model, jt.Module):
            raise TypeError(f"load_model must return jittor.Module, got {type(model)!r}")
        model.eval()
        with jt.no_grad():
            outputs = submission.forward_flow(model, image1, image2)
            if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
                raise TypeError("forward_flow must return (flow_low, flow_up)")
            flow_low, flow_up = outputs
            if not isinstance(flow_low, jt.Var) or not isinstance(flow_up, jt.Var):
                raise TypeError("forward_flow outputs must both be jittor.Var")
            jt.sync_all(True)
            flow_low_array = np.asarray(flow_low.numpy(), dtype=np.float32)
            flow_up_array = np.asarray(flow_up.numpy(), dtype=np.float32)

        expected_low = tuple(config["outputs"]["flow_low_shape"])
        expected_up = tuple(config["outputs"]["flow_up_shape"])
        if flow_low_array.shape != expected_low:
            raise ValueError(f"expected flow_low shape {expected_low}, got {flow_low_array.shape}")
        if flow_up_array.shape != expected_up:
            raise ValueError(f"expected flow_up shape {expected_up}, got {flow_up_array.shape}")
        if not np.isfinite(flow_low_array).all() or not np.isfinite(flow_up_array).all():
            raise ValueError("Jittor flow contains NaN or infinity")

        signed_difference = flow_up_array.astype(np.float64) - reference.astype(np.float64)
        absolute_difference = np.abs(signed_difference)
        per_pixel = np.sqrt(np.sum(np.square(signed_difference), axis=1))
        result["rn"] = True
        result["na"] = bool(
            np.allclose(
                reference,
                flow_up_array,
                rtol=float(config["alignment"]["rtol"]),
                atol=float(config["alignment"]["atol"]),
                equal_nan=bool(config["alignment"]["equal_nan"]),
            )
        )
        result["max_error"] = float(absolute_difference.max())
        result["mean_error"] = float(absolute_difference.mean())
        result["mean_framework_flow_difference"] = float(per_pixel.mean())
        result["max_framework_flow_difference"] = float(per_pixel.max())
        result["p99_framework_flow_difference"] = float(np.percentile(per_pixel, 99))
        result["shape"] = list(flow_up_array.shape)
    except Exception as error:
        result["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started

    result_dir.mkdir(parents=True, exist_ok=True)
    attempt_stem = f"attempt_{attempt_number:02d}"
    write_json(attempt_dir / f"{attempt_stem}.json", result)
    write_json(result_dir / "result.json", result)
    final_flow = result_dir / "flow_up.npy"
    if flow_up_array is not None:
        attempt_flow = attempt_dir / f"{attempt_stem}_flow_up.npy"
        np.save(attempt_flow, flow_up_array, allow_pickle=False)
        shutil.copyfile(attempt_flow, final_flow)
    elif final_flow.exists():
        final_flow.unlink()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["rn"] and result["na"] else 2


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
