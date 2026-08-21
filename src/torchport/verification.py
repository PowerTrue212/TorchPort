from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .parity import compare_tensor_archives


@dataclass(frozen=True)
class ModelVerificationConfig:
    target_script: str
    weights: str


MODEL_VERIFICATION_CONFIGS: dict[str, ModelVerificationConfig] = {
    "cnn-resnet18": ModelVerificationConfig(
        target_script="scripts/run_jittor_resnet18.py",
        weights="artifacts/models/cnn-resnet18/model.safetensors",
    ),
    "vit-tiny-patch16": ModelVerificationConfig(
        target_script="scripts/run_jittor_vit_tiny.py",
        weights="artifacts/models/vit-tiny-patch16/model.safetensors",
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"验证证据不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"验证证据不是有效 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"验证证据必须是 JSON 对象：{path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _find_source_model(source_evidence: dict[str, Any], model_id: str) -> dict[str, Any]:
    models = source_evidence.get("models", [])
    if not isinstance(models, list):
        raise ValueError("源框架证据的 models 字段必须是数组。")
    for model in models:
        if isinstance(model, dict) and model.get("model_id") == model_id:
            return model
    raise ValueError(f"源框架证据中没有模型：{model_id}")


def _tensor_metrics(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    for item in report.get("tensors", []):
        if isinstance(item, dict) and item.get("name") == name:
            return {
                key: item[key]
                for key in (
                    "passed",
                    "max_absolute_error",
                    "mean_absolute_error",
                    "p99_absolute_error",
                    "max_relative_error",
                    "cosine_similarity",
                    "mismatched_elements",
                    "mismatch_ratio",
                )
                if key in item
            }
    return None


def _profile_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(report.get("passed")),
        "rtol": report.get("rtol"),
        "atol": report.get("atol"),
        "first_divergence": report.get("first_divergence"),
        "first_gating_divergence": report.get("first_gating_divergence"),
        "output": _tensor_metrics(report, "output"),
    }


def build_verification_certificate(
    *,
    model_id: str,
    run_root: Path,
    source_evidence: dict[str, Any],
    target_evidence: dict[str, Any],
    fp32_report: dict[str, Any],
    strict_report: dict[str, Any],
) -> dict[str, Any]:
    """把双端运行证据和两档数值报告收敛为统一验证证书。"""
    source_model = _find_source_model(source_evidence, model_id)
    target_model_id = target_evidence.get("model_id")
    if target_model_id != model_id:
        raise ValueError(f"目标证据模型不匹配：期望 {model_id}，实际 {target_model_id}")

    source_input_hash = source_model.get("input", {}).get("sha256")
    target_input_hash = target_evidence.get("input", {}).get("sha256")
    input_identical = (
        source_input_hash == target_input_hash
        if source_input_hash is not None and target_input_hash is not None
        else None
    )
    inference_passed = bool(fp32_report.get("passed"))
    verified = inference_passed and input_identical is not False
    tensor_count = len(fp32_report.get("tensors", []))

    return {
        "schema_version": 1,
        "kind": "torchport.verification-certificate",
        "model_id": model_id,
        "status": "verified" if verified else "failed",
        "passed": verified,
        "run_root": str(run_root.resolve()),
        "source": {
            "framework": source_evidence.get("framework"),
            "framework_version": source_evidence.get("torch_version"),
            "device": source_evidence.get("device"),
            "device_name": source_evidence.get("device_name"),
            "model_class": source_model.get("model_class"),
            "input_sha256": source_input_hash,
            "output_sha256": source_model.get("output", {}).get("sha256"),
        },
        "target": {
            "framework": target_evidence.get("framework"),
            "framework_version": target_evidence.get("jittor_version"),
            "numpy_version": target_evidence.get("numpy_version"),
            "device": target_evidence.get("device"),
            "model_class": target_evidence.get("model_class"),
            "weights_loaded": target_evidence.get("weights", {}).get("loaded"),
            "input_sha256": target_input_hash,
            "output_sha256": target_evidence.get("output", {}).get("sha256"),
        },
        "evidence": {
            "tensor_count": tensor_count,
            "intermediate_probe_count": max(tensor_count - 2, 0),
            "input_identical": input_identical,
            "reference_archive": fp32_report.get("reference"),
            "candidate_archive": fp32_report.get("candidate"),
        },
        "profiles": {
            "fp32-inference": _profile_summary(fp32_report),
            "operator-strict": _profile_summary(strict_report),
        },
    }


class VerificationWorkflow:
    def __init__(
        self,
        project_root: str | Path,
        *,
        python_executable: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.python_executable = str(python_executable or sys.executable)
        if not (self.project_root / "scripts" / "run_pytorch_reference.py").is_file():
            raise ValueError(f"项目目录缺少验证脚本：{self.project_root}")

    def verify(
        self,
        model_id: str,
        *,
        run_root: str | Path,
        source_device: str = "cpu",
        target_device: str = "cpu",
        reuse_existing: bool = False,
    ) -> dict[str, Any]:
        if model_id not in MODEL_VERIFICATION_CONFIGS:
            raise ValueError(f"尚未配置验证路线：{model_id}")
        if source_device not in {"cpu", "cuda"} or target_device not in {"cpu", "cuda"}:
            raise ValueError("设备必须是 cpu 或 cuda。")

        root = Path(run_root).expanduser()
        if not root.is_absolute():
            root = self.project_root / root
        root = root.resolve()
        source_evidence_path = root / "pytorch" / "evidence.json"
        source_tensor_path = root / "pytorch" / "tensors" / f"{model_id}.npz"
        target_evidence_path = root / "jittor" / "evidence.json"
        target_tensor_path = root / "jittor" / f"{model_id}.npz"

        if reuse_existing:
            for path in (
                source_evidence_path,
                source_tensor_path,
                target_evidence_path,
                target_tensor_path,
            ):
                if not path.is_file():
                    raise ValueError(f"--reuse-existing 缺少证据：{path}")
        else:
            self._run_source(
                model_id=model_id,
                device=source_device,
                evidence_path=source_evidence_path,
                tensor_directory=source_tensor_path.parent,
            )
            self._run_target(
                model_id=model_id,
                device=target_device,
                source_tensor_path=source_tensor_path,
                evidence_path=target_evidence_path,
                tensor_path=target_tensor_path,
            )

        fp32_report = compare_tensor_archives(
            source_tensor_path,
            target_tensor_path,
            profile="fp32-inference",
        )
        strict_report = compare_tensor_archives(
            source_tensor_path,
            target_tensor_path,
            profile="operator-strict",
        )
        _write_json(root / "report.json", fp32_report)
        _write_json(root / "report-operator-strict.json", strict_report)

        certificate = build_verification_certificate(
            model_id=model_id,
            run_root=root,
            source_evidence=_read_json(source_evidence_path),
            target_evidence=_read_json(target_evidence_path),
            fp32_report=fp32_report,
            strict_report=strict_report,
        )
        certificate["artifacts"] = {
            "certificate": str((root / "certificate.json").resolve()),
            "fp32_report": str((root / "report.json").resolve()),
            "strict_report": str((root / "report-operator-strict.json").resolve()),
        }
        _write_json(root / "certificate.json", certificate)
        return certificate

    def _pythonpath(self, *relative_paths: str) -> str:
        entries = [str((self.project_root / path).resolve()) for path in relative_paths]
        inherited = os.environ.get("PYTHONPATH")
        if inherited:
            entries.append(inherited)
        return os.pathsep.join(entries)

    def _execute(self, command: list[str], *, pythonpath: str) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = pythonpath
        try:
            subprocess.run(
                command,
                cwd=self.project_root,
                env=environment,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            message = f"验证子进程失败，退出码 {exc.returncode}：{' '.join(command)}"
            raise ValueError(message) from exc

    def _run_source(
        self,
        *,
        model_id: str,
        device: str,
        evidence_path: Path,
        tensor_directory: Path,
    ) -> None:
        self._execute(
            [
                self.python_executable,
                str(self.project_root / "scripts" / "run_pytorch_reference.py"),
                "--device",
                device,
                "--models",
                model_id,
                "--models-root",
                str(self.project_root / "artifacts" / "models"),
                "--output",
                str(evidence_path),
                "--tensors-directory",
                str(tensor_directory),
            ],
            pythonpath=self._pythonpath("artifacts/python-deps", "src"),
        )

    def _run_target(
        self,
        *,
        model_id: str,
        device: str,
        source_tensor_path: Path,
        evidence_path: Path,
        tensor_path: Path,
    ) -> None:
        config = MODEL_VERIFICATION_CONFIGS[model_id]
        self._execute(
            [
                self.python_executable,
                str(self.project_root / config.target_script),
                "--device",
                device,
                "--weights",
                str(self.project_root / config.weights),
                "--input-evidence",
                str(source_tensor_path),
                "--output",
                str(evidence_path),
                "--tensors",
                str(tensor_path),
            ],
            pythonpath=self._pythonpath(
                "artifacts/jittor-deps",
                "artifacts/python-deps",
                "src",
            ),
        )
