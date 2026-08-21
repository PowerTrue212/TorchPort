from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from torchport.parity import compare_tensor_archives

from .cert import build_cert
from .envvals import apply_env
from .merge import merge_probe


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class AlignRun:
    def align(
        self,
        source: str | Path,
        target: str | Path,
        *,
        run_root: str | Path,
        entry: str | None = None,
        model: str | None = None,
        weights: str | Path | None = None,
        inputs: list[dict[str, Any]] | None = None,
        shapes: tuple[tuple[int, ...], ...] | None = None,
        dtypes: tuple[str, ...] | None = None,
        model_args: list[Any] | None = None,
        model_kw: dict[str, Any] | None = None,
        seed: int = 7,
        source_python: str | Path | None = None,
        target_python: str | Path | None = None,
        source_path: Iterable[str | Path] = (),
        target_path: Iterable[str | Path] = (),
    ) -> dict[str, Any]:
        source = Path(source).resolve()
        target = Path(target).resolve()
        root = Path(run_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        spec = self._spec(
            target,
            entry,
            model,
            weights,
            inputs,
            shapes,
            dtypes,
            model_args,
            model_kw,
        )
        source_npz = root / "source.npz"
        target_npz = root / "target.npz"
        state = root / "weights.npz"
        common = [
            "--entry",
            spec["entry"],
            "--model",
            spec["model"],
            "--model-args",
            json.dumps(spec["args"]),
            "--model-kw",
            json.dumps(spec["kwargs"]),
            "--seed",
            str(seed),
            "--state",
            str(state),
        ]
        for item in spec["inputs"]:
            common.extend(["--input", json.dumps(item)])
        source_args = ["capture", "source", str(source), *common]
        if spec["weights"]:
            source_args.extend(["--weights", spec["weights"]])
        source_result = self._run(
            str(source_python or sys.executable),
            [*source_args, "--output", str(source_npz)],
            source_path,
            target=False,
            runtime=spec["env"],
        )
        self._ensure(source_npz, "源端", source_result)
        target_result = self._run(
            str(target_python or sys.executable),
            [
                "capture",
                "target",
                str(target),
                *common,
                "--output",
                str(target_npz),
                "--source",
                str(source_npz),
            ],
            target_path,
            target=True,
            runtime=spec["env"],
        )
        self._ensure(target_npz, "目标端", target_result)
        fp32 = compare_tensor_archives(source_npz, target_npz, profile="fp32-inference")
        strict = compare_tensor_archives(source_npz, target_npz, profile="operator-strict")
        _write(root / "report.json", fp32)
        _write(root / "strict.json", strict)
        source_info = _read(source_npz.with_suffix(".json"))
        target_info = _read(target_npz.with_suffix(".json"))
        result = build_cert(
            str(source), str(target), spec, source_info, target_info, fp32, strict
        )
        result["artifacts"] = {
            "source": str(source_npz),
            "target": str(target_npz),
            "weights": str(state),
            "report": str(root / "report.json"),
            "strict": str(root / "strict.json"),
        }
        _write(root / "certificate.json", result)
        return result

    @staticmethod
    def _spec(
        target: Path,
        entry: str | None,
        model: str | None,
        weights: str | Path | None,
        inputs: list[dict[str, Any]] | None,
        shapes: tuple[tuple[int, ...], ...] | None,
        dtypes: tuple[str, ...] | None,
        model_args: list[Any] | None,
        model_kw: dict[str, Any] | None,
    ) -> dict[str, Any]:
        path = target / ".torchport" / "probe.json"
        saved = _read(path) if path.is_file() else {}
        selected = str(weights) if weights is not None else saved.get("weights")
        if selected is None and saved.get("weight_candidates"):
            names = ", ".join(saved["weight_candidates"])
            raise ValueError(f"发现多个检查点，请使用 --weights 选择：{names}")
        spec = merge_probe(
            saved,
            entry=entry,
            model=model,
            inputs=inputs,
            shapes=shapes,
            dtypes=dtypes,
            model_args=model_args,
            model_kw=model_kw,
        )
        spec["weights"] = selected
        missing = [key for key in ("entry", "model", "inputs") if not spec[key]]
        if missing:
            raise ValueError(f"探针缺少配置：{', '.join(missing)}")
        return spec

    @staticmethod
    def _run(
        python: str,
        args: list[str],
        paths: Iterable[str | Path],
        *,
        target: bool,
        runtime: dict[str, str | None],
    ) -> subprocess.CompletedProcess[str]:
        package = Path(__file__).resolve().parents[2]
        entries = [str(package), *(str(Path(path).resolve()) for path in paths)]
        inherited = os.environ.get("PYTHONPATH")
        if inherited:
            entries.append(inherited)
        env = os.environ.copy()
        apply_env(env, runtime)
        env["PYTHONPATH"] = os.pathsep.join(entries)
        if target:
            env["nvcc_path"] = ""
        result = subprocess.run(
            [python, "-m", "torchport", *args],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            detail = result.stderr[-3000:] or result.stdout[-3000:]
            side = "目标端" if target else "源端"
            raise ValueError(f"{side}探针子进程失败：{detail.strip()}")
        return result

    @staticmethod
    def _ensure(
        path: Path,
        side: str,
        result: subprocess.CompletedProcess[str] | None,
    ) -> None:
        if path.is_file():
            return
        detail = ""
        if result is not None:
            detail = result.stderr[-3000:] or result.stdout[-3000:]
        raise ValueError(f"{side}探针未生成证据：{path}；{detail.strip()}")
