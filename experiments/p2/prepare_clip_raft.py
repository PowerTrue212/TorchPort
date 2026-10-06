"""Materialize identical CLIP/RAFT P2 workspaces and public/hidden TaskSpecs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("verdict", "diagnostic", "localized")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8")


def _copy_seed(task: str, destination: Path) -> None:
    source = (
        ROOT / task.upper() / "source" / ("openai-clip" if task == "clip" else "princeton-vl-raft")
    )
    if not source.is_dir():
        raise FileNotFoundError(f"先准备冻结的 {task} source：{source}")
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / task.upper() / "templates" / "submission.py", destination / "submission.py")
    shutil.copy2(
        ROOT / "benchmarks" / "tasks" / task / "target_adapter.py",
        destination / "torchport_adapter.py",
    )


def _hidden_inputs(task: str, private: Path) -> list[Path]:
    prepared = ROOT / task.upper() / "shared" / "prepared"
    private.mkdir(parents=True, exist_ok=True)
    if task == "clip":
        values = [
            np.load(prepared / name, allow_pickle=False) for name in ("images.npy", "tokens.npy")
        ]
        names = ("images.npy", "tokens.npy")
        hidden = [value[::-1].copy() for value in values]
    else:
        values = [
            np.load(prepared / name, allow_pickle=False) for name in ("image1.npy", "image2.npy")
        ]
        names = ("image1.npy", "image2.npy")
        hidden = [values[1].copy(), values[0].copy()]
    paths = []
    for name, value in zip(names, hidden):
        path = private / name
        np.save(path, value, allow_pickle=False)
        paths.append(path)
    return paths


def _input(path: Path, *, dtype: str) -> dict[str, Any]:
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    return {"path": str(path.resolve()), "shape": list(value.shape), "dtype": dtype}


def _target_env(source_python: Path) -> dict[str, str]:
    env = {"python_config_path": str(source_python.with_name("python3.12-config"))}
    system_cpp = Path("/usr/lib/x86_64-linux-gnu")
    system_libstdcpp = system_cpp / "libstdc++.so.6"
    if system_libstdcpp.exists():
        inherited = os.environ.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{system_cpp}:{inherited}" if inherited else str(system_cpp)
        env["LD_PRELOAD"] = str(system_libstdcpp)
    env["conv_opt"] = "1"
    env["use_cutlass"] = "0"
    return env


def _task_spec(
    task: str,
    workspace: Path,
    inputs: list[Path],
    source_python: Path,
    target_python: Path,
) -> dict[str, Any]:
    base = ROOT / task.upper()
    weights = base / "shared" / "prepared" / "weights_fp32.npz"
    source_child = base / "source" / ("openai-clip" if task == "clip" else "princeton-vl-raft")
    dtypes = ("float32", "int64") if task == "clip" else ("float32", "float32")
    timeout = 900 if task == "clip" else 1800
    return {
        "schema_version": 1,
        "task_id": f"{task}-p2",
        "source": {
            "repo": str((base / "source").resolve()),
            "entry": "torchport_adapter.py",
            "model": "TorchPortCLIP" if task == "clip" else "TorchPortRAFT",
            "python": str(source_python.absolute()),
            "device": "cuda",
            "paths": [str(source_child.resolve())]
            if task == "clip"
            else [str((source_child / "core").resolve())],
        },
        "target": {
            "repo": str(workspace.resolve()),
            "entry": "torchport_adapter.py",
            "model": "TorchPortCLIP" if task == "clip" else "TorchPortRAFT",
            "kwargs": {"weights_path": str(weights.resolve())},
            "python": str(target_python.absolute()),
            "device": "cuda",
            "env": _target_env(source_python),
        },
        "inputs": [_input(path, dtype=dtype) for path, dtype in zip(inputs, dtypes)],
        "weights": str(weights.resolve()),
        "seed": 0,
        "protocol": {"profile": "fp32-inference", "repeats": 3, "timeout": timeout},
    }


def materialize(
    output: Path,
    source_python: Path,
    target_python: Path,
    replicates: int,
) -> dict[str, Any]:
    template_path = ROOT / "experiments" / "p2" / "clip-raft-pilot" / "experiment.json"
    experiment = json.loads(template_path.read_text("utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    prompt = output / "prompt.md"
    shutil.copy2(ROOT / "experiments" / "harness" / "prompt.example.md", prompt)
    experiment["prompt"] = prompt.name
    trials = []
    hidden_by_task = {
        task: _hidden_inputs(task, output / "private" / task) for task in ("clip", "raft")
    }
    for task in ("clip", "raft"):
        prepared = ROOT / task.upper() / "shared" / "prepared"
        public_inputs = (
            [prepared / "images.npy", prepared / "tokens.npy"]
            if task == "clip"
            else [prepared / "image1.npy", prepared / "image2.npy"]
        )
        for condition in CONDITIONS:
            for replicate in range(replicates):
                trial_id = f"{task}-{condition}-r{replicate}"
                workspace = output / "workspaces" / trial_id / "repo"
                if workspace.exists():
                    raise FileExistsError(f"拒绝覆盖已有 trial workspace：{workspace}")
                _copy_seed(task, workspace)
                public_task = _task_spec(
                    task, workspace, public_inputs, source_python, target_python
                )
                hidden_task = _task_spec(
                    task, workspace, hidden_by_task[task], source_python, target_python
                )
                task_dir = output / "tasks" / trial_id
                _write(task_dir / "public.json", public_task)
                _write(task_dir / "hidden.json", hidden_task)
                trials.append(
                    {
                        "trial_id": trial_id,
                        "task": task,
                        "condition": condition,
                        "replicate": replicate,
                        "path": f"runs/{task}/{condition}-r{replicate}",
                        "public_task": f"tasks/{trial_id}/public.json",
                        "hidden_task": f"tasks/{trial_id}/hidden.json",
                    }
                )
    experiment["replicates"] = replicates
    experiment["trials"] = trials
    experiment["frozen_artifacts"] = {
        str(path.relative_to(output).as_posix()): _sha(path)
        for path in [prompt, *sorted((output / "private").rglob("*.npy"))]
    }
    _write(output / "experiment.json", experiment)
    return experiment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-python", type=Path, required=True)
    parser.add_argument("--target-python", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1)
    args = parser.parse_args(argv)
    if args.replicates < 1:
        parser.error("--replicates 必须为正数")
    report = materialize(
        args.output.resolve(), args.source_python, args.target_python, args.replicates
    )
    print(
        json.dumps(
            {"experiment_id": report["experiment_id"], "trials": len(report["trials"])}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
