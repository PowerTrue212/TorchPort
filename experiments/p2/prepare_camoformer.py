"""Materialize isolated CamoFormer COD trials from one frozen source seed."""

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
SOURCE_REVISION = "5f924a91a13707b71ca1b986924504464ffd0638"
REFERENCE_REVISION = "be601810aa1e249440a5da281101bdfaae898cba"
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


def _revision(repo: Path) -> str:
    head = repo / ".git" / "HEAD"
    if not head.is_file():
        raise ValueError(f"缺少冻结 Git checkout：{repo}")
    import subprocess

    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return value


def _target_env(source_python: Path) -> dict[str, str]:
    env = {"python_config_path": str(source_python.with_name("python3.12-config"))}
    system_cpp = Path("/usr/lib/x86_64-linux-gnu")
    system_libstdcpp = system_cpp / "libstdc++.so.6"
    if system_libstdcpp.exists():
        inherited = os.environ.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{system_cpp}:{inherited}" if inherited else str(system_cpp)
        env["LD_PRELOAD"] = str(system_libstdcpp)
    env.update(conv_opt="1", use_cutlass="0")
    return env


def _input(path: Path) -> dict[str, Any]:
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    return {"path": str(path.resolve()), "shape": list(value.shape), "dtype": str(value.dtype)}


def _task(
    source: Path,
    target: Path,
    input_path: Path,
    source_python: Path,
    target_python: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": "camoformer-cod-pilot",
        "source": {
            "repo": str(source.resolve()),
            "entry": "torchport_adapter.py",
            "model": "TorchPortCamoFormer",
            "python": str(source_python.absolute()),
            "device": "cuda",
        },
        "target": {
            "repo": str(target.resolve()),
            "entry": "torchport_adapter.py",
            "model": "TorchPortCamoFormer",
            "python": str(target_python.absolute()),
            "device": "cuda",
            "env": _target_env(source_python),
        },
        "inputs": [_input(input_path)],
        "seed": 20261004,
        "protocol": {"profile": "fp32-inference", "repeats": 3, "timeout": 1800},
    }


def materialize(
    output: Path,
    source_python: Path,
    target_python: Path,
    replicates: int,
    conditions: Sequence[str] = CONDITIONS,
    experiment_id: str = "p2-camoformer-cod-v41-final-20261004",
) -> dict[str, Any]:
    source = ROOT / "CamoFormer" / "source" / "HVision-NKU-CamoFormer"
    reference = ROOT / "CamoFormer" / "reference" / "CamoFormer-Jittor"
    if _revision(source) != SOURCE_REVISION:
        raise ValueError(f"CamoFormer source revision 必须为 {SOURCE_REVISION}")
    if reference.exists() and _revision(reference) != REFERENCE_REVISION:
        raise ValueError(f"CamoFormer reference revision 必须为 {REFERENCE_REVISION}")
    shutil.copy2(
        ROOT / "benchmarks" / "tasks" / "camoformer" / "source_adapter.py",
        source / "torchport_adapter.py",
    )

    fixtures = output / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    prompt = output / "prompt.md"
    shutil.copy2(ROOT / "CamoFormer" / "prompt-agent-zh.md", prompt)
    rng = np.random.default_rng(20261004)
    public_input = fixtures / "public-image.npy"
    hidden_input = fixtures / "hidden-image.npy"
    np.save(public_input, rng.standard_normal((1, 3, 128, 128)).astype(np.float32))
    np.save(hidden_input, rng.standard_normal((1, 3, 128, 128)).astype(np.float32))

    trials = []
    for condition in conditions:
        for replicate in range(replicates):
            trial_id = f"camoformer-{condition}-r{replicate}"
            workspace = output / "workspaces" / trial_id / "repo"
            if workspace.exists():
                raise FileExistsError(f"拒绝覆盖已有 trial workspace：{workspace}")
            workspace.mkdir(parents=True)
            shutil.copy2(ROOT / "CamoFormer" / "templates" / "submission.py", workspace)
            shutil.copy2(
                ROOT / "benchmarks" / "tasks" / "camoformer" / "target_adapter.py",
                workspace / "torchport_adapter.py",
            )
            public = _task(source, workspace, public_input, source_python, target_python)
            hidden = _task(source, workspace, hidden_input, source_python, target_python)
            task_dir = output / "tasks" / trial_id
            _write(task_dir / "public.json", public)
            _write(task_dir / "hidden.json", hidden)
            trials.append(
                {
                    "trial_id": trial_id,
                    "task": "camoformer",
                    "condition": condition,
                    "replicate": replicate,
                    "path": f"runs/camoformer/{condition}-r{replicate}",
                    "public_task": f"tasks/{trial_id}/public.json",
                    "hidden_task": f"tasks/{trial_id}/hidden.json",
                }
            )

    experiment = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "purpose": "Exercise the controlled migration harness on a complete COD forward pass.",
        "provider": "deepseek",
        "model": "deepseek-flash",
        "model_version": "DeepSeek-V4.1-Flash",
        "temperature": 0.0,
        "thinking": "enabled",
        "reasoning_effort": "max",
        "network_research": True,
        "prompt": prompt.name,
        "runtime_packages": ["timm", "einops", "opencv-python-headless"],
        "hardware": {"gpu": "frozen by runtime-snapshot.json", "execution": "serial"},
        "budget": {
            "oracle_queries": 20,
            "wall_seconds": 7200,
            "total_tokens": 800000,
            "model_turns": 100,
        },
        "replicates": replicates,
        "tasks": ["camoformer"],
        "conditions": list(conditions),
        "isolation": {
            "level": "trusted_candidate_pilot",
            "network_isolation": False,
            "host_filesystem_isolation": False,
            "formal_claims_allowed": False,
            "reason": "First COD integration smoke test; formal worker is not yet available.",
        },
        "upstream": {
            "source": {
                "url": "https://github.com/HVision-NKU/CamoFormer",
                "revision": SOURCE_REVISION,
            },
            "calibration_reference": {
                "url": "https://github.com/yinbow/CamoFormer-Jittor",
                "revision": REFERENCE_REVISION,
                "agent_visible": False,
            },
        },
        "price_snapshot": None,
        "frozen_artifacts": {
            path.relative_to(output).as_posix(): _sha(path)
            for path in (prompt, public_input, hidden_input)
        },
        "trials": trials,
    }
    _write(output / "experiment.json", experiment)
    return experiment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-python", type=Path, required=True)
    parser.add_argument("--target-python", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    parser.add_argument("--experiment-id", default="p2-camoformer-cod-v41-final-20261004")
    args = parser.parse_args(argv)
    if args.replicates < 1:
        parser.error("--replicates 必须为正数")
    report = materialize(
        args.output.resolve(),
        args.source_python,
        args.target_python,
        args.replicates,
        args.conditions,
        args.experiment_id,
    )
    print(
        json.dumps(
            {"experiment_id": report["experiment_id"], "trials": len(report["trials"])}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
