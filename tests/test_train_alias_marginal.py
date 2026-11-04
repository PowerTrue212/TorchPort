"""Shared-module canonical names and train-step marginal rerun (2026-11-04)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from torchport.capture import training
from torchport.comparison import compare_noise_calibrated
from torchport.contracts import normalize
from torchport.contracts.identity import oracle_key
from torchport.verification import service


class Parameter:
    def __init__(self) -> None:
        self.shape = (1,)
        self.requires_grad = True


class Optimizer:
    def __init__(self, params) -> None:
        self.param_groups = [{"params": list(params), "lr": 1e-3}]


class SharedModel:
    """Jittor style: a shared module yields its parameter once per name."""

    def __init__(self) -> None:
        self.shared = Parameter()
        self.other = Parameter()

    def named_parameters(self):
        return [
            ("layer.norm3.weight", self.shared),
            ("layer.downsample.1.weight", self.shared),
            ("layer.conv.weight", self.other),
        ]


def test_shared_parameter_takes_first_registered_name() -> None:
    model = SharedModel()
    trainable = training._trainable(model, Optimizer([model.shared, model.other]))
    assert list(trainable) == ["layer.conv.weight", "layer.norm3.weight"]


def _task(root: Path, **protocol) -> dict:
    inputs = root / "x.npy"
    np.save(inputs, np.zeros((2, 1), np.float32))
    for side in ("source", "target"):
        (root / side).mkdir(exist_ok=True)
        (root / side / "entry.py").write_text("x = 1\n")
    return normalize({
        "task_id": "t",
        "source": {"repo": str(root / "source"), "entry": "entry.py", "model": "m"},
        "target": {"repo": str(root / "target"), "entry": "entry.py", "model": "m"},
        "inputs": [{"shape": [2, 1], "dtype": "float32", "path": str(inputs)}],
        "protocol": {"profile": "train-step", "repeats": 2,
                     "train": {"optimizer": "o", "step": "s"}, **protocol},
    }, root)


def test_marginal_rerun_validation_and_oracle_identity() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        plain = _task(root)
        enabled = _task(root, marginal_rerun={"upper": 3.0, "runs": 5})
        assert enabled["protocol"]["marginal_rerun"] == {"upper": 3.0, "runs": 5}
        # judge-only: enabling the rerun must not invalidate the source oracle
        assert oracle_key(plain) == oracle_key(enabled)
        for bad in ({"upper": 1.0, "runs": 5}, {"upper": 3.0, "runs": 4},
                    {"upper": 3.0, "runs": 1}, {"upper": 3.0, "extra_runs": 4}):
            with pytest.raises(ValueError):
                _task(root, marginal_rerun=bad)


def test_usage_includes_k_and_floor() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        key = "output.loss.0"
        for name, value in (("source", 1.0), ("fp64", 1.0 + 1e-6), ("alt", 1.0)):
            np.savez(root / f"{name}.npz", **{key: np.asarray([value])})
        np.savez(root / "cand.npz", **{key: np.asarray([1.0 + 1e-6 + 1.6e-5])})
        report = compare_noise_calibrated(
            root / "source.npz", root / "cand.npz", root / "fp64.npz", root / "alt.npz",
            require_calibrated=True,
        )
        metrics = report["metrics"]
        # err 1.6e-5, limit 8 * 1e-6 + 1e-7 = 8.1e-6 -> usage ~1.98
        assert metrics["max_gating_usage"] == pytest.approx(1.6e-5 / 8.1e-6, rel=1e-3)
        assert metrics["numeric_only_failures"] is True
        assert not report["passed"]


def _result(usage: float | None, *, numeric: bool = True) -> dict:
    return {
        "passed": usage is not None and usage <= 1,
        "runnable": "pass",
        "structural": "pass" if numeric else "fail",
        "failure_code": None if usage is not None and usage <= 1 else "PARITY_FAILED",
        "train_exact": {"passed": True},
        "envelope_metrics": {"max_gating_usage": usage, "numeric_only_failures": numeric},
    }


def _rerun(monkeypatch, first: list[dict], extra: list[dict]) -> dict | None:
    queue = list(extra)
    monkeypatch.setattr(service, "run_side", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(service, "evaluate", lambda *a, **k: queue.pop(0))
    task = {"protocol": {}, "target_identity": "x"}
    return service._marginal_rerun(task, Path("."), "t", {"dir": "."}, list(first), [],
                                   {"upper": 3.0, "runs": 5})


def test_marginal_median_passes_and_fails(monkeypatch) -> None:
    good = _rerun(monkeypatch, [_result(1.4)], [_result(u) for u in (0.5, 0.6, 0.7, 1.2)])
    assert good["passed"] and good["median_usage"] == pytest.approx(0.7)
    bad = _rerun(monkeypatch, [_result(1.4)], [_result(u) for u in (1.1, 1.3, 0.5, 0.6)])
    assert not bad["passed"] and bad["median_usage"] == pytest.approx(1.1)


def test_marginal_not_triggered_outside_band_or_on_structure(monkeypatch) -> None:
    assert _rerun(monkeypatch, [_result(3.5)], []) is None
    assert _rerun(monkeypatch, [_result(1.4, numeric=False)], []) is None


def test_marginal_rerun_stops_on_non_numeric_failure(monkeypatch) -> None:
    report = _rerun(monkeypatch, [_result(1.4)],
                    [_result(0.5), _result(None, numeric=False)])
    assert not report["passed"] and report["runs"] == 3 and report["median_usage"] is None
