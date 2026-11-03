"""Frozen reference repeat noise is judge-only and cannot hide training errors."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from torchport.comparison import compare_noise_calibrated
from torchport.contracts import normalize
from torchport.contracts.identity import file_hash
from torchport.verification.noise import _rel_spread


def _archive(path: Path, loss: float, norm: float, update: float, summed: float) -> Path:
    np.savez(
        path,
        **{
            "output.loss.0": np.asarray([loss]),
            "output.ustat.0.w": np.asarray([norm, 0.1, summed]),
            "output.update.0.w": np.asarray([update, 0.02]),
        },
    )
    return path


def test_fixed_repeat_baseline_passes_reference_and_rejects_two_errors() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = _archive(root / "source.npz", 1.0, 1.0, 0.01, 100.0)
        fp64 = _archive(root / "fp64.npz", 1.0, 1.0, 0.01, 100.0)
        alt = _archive(root / "alt.npz", 1.0, 1.0, 0.01, 100.0)
        good = _archive(root / "good.npz", 1.0 + 3e-6, 1.0 + 3e-5, 0.01 + 3e-5, 1e6)
        wrong_optimizer = _archive(root / "optimizer.npz", 1.0 + 1e-3, 1.0 + 3e-5, 0.01 + 1e-3, 1e6)
        wrong_temperature = _archive(root / "temperature.npz", 1.0 + 1e-2, 1.0 + 3e-5, 0.01, 1e6)
        baseline = root / "baseline.npz"
        np.savez(baseline, **{
            "output.loss.0": np.asarray([2e-6, 2e-6]),
            "output.ustat.0.w": np.asarray([2e-5, 2e-5]),
            "output.update.0.w": np.asarray([2e-5, 2e-5]),
        })

        assert not compare_noise_calibrated(source, good, fp64, alt)["passed"]
        options = {"target_baseline_path": baseline, "drop_ustat_sum": True, "require_calibrated": True}
        accepted = compare_noise_calibrated(source, good, fp64, alt, **options)
        assert accepted["passed"]
        assert accepted["profile"] == "train-repeat-envelope-v1-pilot"
        assert not compare_noise_calibrated(source, wrong_optimizer, fp64, alt, **options)["passed"]
        assert not compare_noise_calibrated(source, wrong_temperature, fp64, alt, **options)["passed"]

        missing = root / "missing.npz"
        np.savez(missing, **{"output.loss.0": np.asarray([2e-6, 2e-6])})
        report = compare_noise_calibrated(
            source, good, fp64, alt,
            target_baseline_path=missing, drop_ustat_sum=True, require_calibrated=True,
        )
        assert not report["passed"]
        assert "output.ustat.0.w" in report["uncalibrated"]


def test_complex_spectral_weight_imaginary_error_is_not_discarded() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        key = "output.param.0.spectral_weight"
        source = root / "source.npz"
        fp64 = root / "fp64.npz"
        alt = root / "alt.npz"
        candidate = root / "candidate.npz"
        for path in (source, fp64, alt):
            np.savez(path, **{key: np.asarray([1 + 2j], dtype=np.complex64)})
        np.savez(candidate, **{key: np.asarray([1 + 3j], dtype=np.complex64)})
        report = compare_noise_calibrated(
            source, candidate, fp64, alt, require_calibrated=True,
        )
        assert not report["passed"]
        assert report["tensors"][0]["candidate_max_abs"] == 1.0
        assert _rel_spread([
            np.asarray([1 + 2j]), np.asarray([1 + 3j]),
        ]) > 0


def test_baseline_hash_is_required_and_does_not_change_source_oracle() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for side in ("source", "target"):
            (root / side).mkdir()
            (root / side / "run.py").write_text("x = 1\n", encoding="utf-8")
        baseline = root / "baseline.npz"
        np.savez(baseline, **{"output.loss.0": np.asarray([1e-6, 1e-6])})
        raw = {
            "task_id": "train-repeat-test",
            "source": {"repo": "source", "entry": "run.py", "model": "load"},
            "target": {"repo": "target", "entry": "run.py", "model": "load"},
            "inputs": [{"shape": [1, 1]}],
            "protocol": {"profile": "train-step", "train": {"optimizer": "o", "step": "s"}},
        }
        old = normalize(raw, root)
        raw["protocol"].update({
            "train_envelope": "train-repeat-envelope-v1-pilot",
            "train_noise_baseline": str(baseline),
            "train_noise_baseline_sha256": file_hash(baseline),
        })
        new = normalize(raw, root)
        assert new["digest"] == old["digest"]
        assert new["protocol"]["train_noise_baseline"] == str(baseline)
        raw["protocol"]["train_noise_baseline_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="SHA256"):
            normalize(raw, root)
