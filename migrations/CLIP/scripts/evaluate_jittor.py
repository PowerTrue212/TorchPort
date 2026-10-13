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
from common import ROOT, load_config, method_ids, read_json_lines, verify_prepared_files, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one migrated Jittor CLIP submission")
    parser.add_argument("--method", required=True, choices=method_ids())
    return parser.parse_args()


def import_submission(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("clip_migration_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import submission: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("load_model", "forward_logits"):
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
    verify_prepared_files()
    rounds_path = ROOT / "results" / args.method / "rounds.jsonl"
    rounds = read_json_lines(rounds_path)
    maximum = int(config["execution"]["max_repair_rounds"])
    if len(rounds) > maximum:
        raise RuntimeError(f"repair log exceeds frozen maximum {maximum}")

    result = base_result(args.method, len(rounds), config)
    result_dir = ROOT / "results" / args.method
    attempt_dir = result_dir / "attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    attempt_number = len(rounds)
    started = time.perf_counter()
    logits_array: np.ndarray | None = None

    try:
        import jittor as jt

        expected_jittor = config["environment"]["jittor"]
        if jt.__version__ != expected_jittor:
            raise RuntimeError(
                f"frozen protocol requires Jittor {expected_jittor}, got {jt.__version__}"
            )
        expected_numpy = config["environment"]["numpy"]
        if np.__version__ != expected_numpy:
            raise RuntimeError(
                f"frozen protocol requires NumPy {expected_numpy}, got {np.__version__}"
            )
        if config["execution"]["device"] != "cuda":
            raise RuntimeError("only the frozen CUDA protocol is supported")
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
        images = np.load(prepared / "images.npy", allow_pickle=False)
        tokens = np.load(prepared / "tokens.npy", allow_pickle=False)
        reference = np.load(prepared / "logits_ref.npy", allow_pickle=False)
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
            logits = submission.forward_logits(model, images, tokens)
            if not isinstance(logits, jt.Var):
                raise TypeError(f"forward_logits must return jittor.Var, got {type(logits)!r}")
            jt.sync_all(True)
            logits_array = np.asarray(logits.numpy(), dtype=np.float32)

        if logits_array.shape != (4, 4):
            raise ValueError(f"expected logits shape (4, 4), got {logits_array.shape}")
        if not np.isfinite(logits_array).all():
            raise ValueError("Jittor logits contain NaN or infinity")

        difference = np.abs(reference.astype(np.float64) - logits_array.astype(np.float64))
        result["rn"] = True
        result["na"] = bool(
            np.allclose(
                reference,
                logits_array,
                rtol=float(config["alignment"]["rtol"]),
                atol=float(config["alignment"]["atol"]),
                equal_nan=bool(config["alignment"]["equal_nan"]),
            )
        )
        result["max_error"] = float(difference.max())
        result["shape"] = list(logits_array.shape)
    except Exception as error:
        result["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started

    attempt_stem = f"attempt_{attempt_number:02d}"
    attempt_json = attempt_dir / f"{attempt_stem}.json"
    write_json(attempt_json, result)
    write_json(result_dir / "result.json", result)
    if logits_array is not None:
        attempt_logits = attempt_dir / f"{attempt_stem}_logits.npy"
        np.save(attempt_logits, logits_array, allow_pickle=False)
        shutil.copyfile(attempt_logits, result_dir / "logits.npy")
    elif (result_dir / "logits.npy").exists():
        (result_dir / "logits.npy").unlink()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["rn"] and result["na"] else 2


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
