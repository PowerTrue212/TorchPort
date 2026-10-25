from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.harness.reporting.benchmark_v1 import score_smoke


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _inputs(root: Path) -> dict:
    trial = _write(root / "result.json", {
        "trial_id": "pilot-r1",
        "claim_status": "pilot_only",
        "valid": False,
        "passed": True,
    })
    official = _write(root / "official.json", {
        "inference": {"passed": True, "results": [{"runnable": "pass"}]},
        "training": {"passed": True, "results": [{"runnable": "pass"}]},
    })
    layer = _write(root / "layer.json", {
        "status": "exploratory_not_formal",
        "checkpoint_count": 2,
        "score": 75.0,
        "rows": [{"score": 1.0}, {"score": 0.5}],
    })
    task = _write(root / "task.json", {
        "provenance": "synthetic_smoke",
        "metrics": [{"name": "quality", "source": 0.8, "target": 0.79, "tau": 0.02}],
    })
    baselines = _write(root / "normalizers.json", {
        "provenance": "synthetic_smoke", "T0": 100, "N0": 1000, "K0": 10, "F0": 2,
    })
    events = root / "events.jsonl"
    events.write_text("\n".join(json.dumps(item) for item in [
        {"event": "agent_session_started"},
        {"event": "work_interval", "category": "migration", "started_at": 0, "finished_at": 60},
        {"event": "work_interval", "category": "compilation", "started_at": 60, "finished_at": 90},
        {"event": "model_response", "usage": {"total_tokens": 500}},
        {"event": "agent_tool_call", "tool": "shell"},
        {"event": "agent_tool_call", "tool": "verify"},
        {"event": "candidate_revision", "digest": "a"},
        {"event": "candidate_revision", "digest": "b"},
        {"event": "revision_check", "digest": "a", "passed": False},
        {"event": "revision_check", "digest": "b", "passed": True},
        {"event": "agent_session_finished"},
    ]) + "\n", encoding="utf-8")
    return {
        "trial_result": trial,
        "official_report": official,
        "layer_score": layer,
        "task_metrics": task,
        "agent_events": events,
        "normalizers": baselines,
    }


def test_four_metric_smoke_uses_evidence_and_never_claims_formal_success(tmp_path: Path) -> None:
    result = score_smoke(**_inputs(tmp_path))
    metrics = result["metrics"]
    assert result["formal_eligible"] is False
    assert result["valid"] is False
    assert metrics["R_e"]["value"] == 1
    assert metrics["B_l"]["value"] == 75.0
    assert metrics["P_t"]["value"] == 100.0
    assert metrics["P_t"]["s"] == 1
    assert metrics["E_m"]["T"] == 60
    assert metrics["E_m"]["N"] == 500
    assert metrics["E_m"]["K"] == 2
    assert metrics["E_m"]["F"] == 1
    assert metrics["E_m"]["C"] == pytest.approx(0.51)
    assert metrics["E_m"]["value"] == pytest.approx(100 / 1.51)


def test_missing_p_or_agent_cost_is_undefined_not_zero(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["task_metrics"] = None
    result = score_smoke(**inputs)
    assert result["metrics"]["P_t"] == {
        "value": None, "status": "undefined", "reason": "task_metrics_missing",
    }
    assert result["metrics"]["E_m"]["value"] is None
    assert result["metrics"]["E_m"]["reason"] == "task_success_undefined"


def test_missing_layer_trace_is_undefined_not_zero(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["layer_score"] = None
    result = score_smoke(**inputs)
    assert result["metrics"]["B_l"] == {
        "value": None, "status": "undefined", "reason": "layer_evidence_missing",
    }


def test_task_metric_failure_sets_s_and_efficiency_to_zero(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    _write(inputs["task_metrics"], {
        "provenance": "synthetic_smoke",
        "metrics": [{"name": "quality", "source": 0.8, "target": 0.7, "tau": 0.02}],
    })
    result = score_smoke(**inputs)
    assert result["metrics"]["P_t"]["value"] == pytest.approx(20.0)
    assert result["metrics"]["P_t"]["s"] == 0
    assert result["metrics"]["E_m"]["value"] == 0


def test_runnable_does_not_mean_numerically_aligned(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    _write(inputs["trial_result"], {
        "trial_id": "pilot", "claim_status": "pilot_only", "valid": False, "passed": False,
    })
    _write(inputs["official_report"], {
        "inference": {"passed": False, "results": [{"runnable": "pass"}]},
        "training": {"passed": False, "results": [{"runnable": "pass"}]},
    })
    result = score_smoke(**inputs)
    assert result["metrics"]["R_e"]["value"] == 1
    assert result["metrics"]["R_e"]["verify_passed"] is False


def test_refuses_formal_trial_and_incomplete_revision_checks(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    trial = _write(inputs["trial_result"], {
        "trial_id": "formal", "claim_status": "formal", "valid": True, "passed": True,
    })
    with pytest.raises(ValueError, match="pilot_only"):
        score_smoke(**{**inputs, "trial_result": trial})

    _write(trial, {
        "trial_id": "pilot", "claim_status": "pilot_only", "valid": False, "passed": True,
    })
    events = inputs["agent_events"]
    lines = events.read_text(encoding="utf-8").splitlines()
    events.write_text("\n".join(line for line in lines if '"digest": "b", "passed"' not in line),
                      encoding="utf-8")
    with pytest.raises(ValueError, match="事件不完整"):
        score_smoke(**inputs)
