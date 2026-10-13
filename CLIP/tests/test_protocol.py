from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return json.loads((ROOT / "experiment.json").read_text(encoding="utf-8"))


def test_frozen_protocol_has_exactly_three_methods_and_four_inputs() -> None:
    value = config()
    assert [item["id"] for item in value["methods"]] == [
        "codex",
        "dsh",
        "dsh_torchport",
    ]
    assert [item["label"] for item in value["input"]["samples"]] == [
        "dog",
        "cat",
        "car",
        "airplane",
    ]
    assert value["input"]["images_shape"] == [4, 3, 224, 224]
    assert value["input"]["tokens_shape"] == [4, 77]


def test_only_dsh_torchport_can_use_torchport() -> None:
    permissions = {item["id"]: item["torchport_allowed"] for item in config()["methods"]}
    assert permissions == {"codex": False, "dsh": False, "dsh_torchport": True}


def test_primary_metrics_and_round_limit_are_frozen() -> None:
    value = config()
    assert value["execution"]["precision"] == "float32"
    assert value["execution"]["max_repair_rounds"] == 8
    assert value["alignment"] == {"rtol": 1e-4, "atol": 1e-4, "equal_nan": False}
    assert value["environment"]["source_python"] == "3.10"
    assert value["environment"]["target_python"] == "3.9"
    assert value["environment"]["pytorch"] == "2.6.0+cu124"
    assert value["environment"]["jittor"] == "1.3.11.0"
    assert value["environment"]["numpy"] == "1.26.4"


def test_prompts_enforce_scope_and_common_evaluator() -> None:
    for method in ("codex", "dsh", "dsh_torchport"):
        prompt = (ROOT / "prompts" / f"{method}.md").read_text(encoding="utf-8")
        assert "run_evaluate.sh" in prompt
        assert "record_round.py" in prompt
        assert "No training, FP16" in prompt
    for method in ("codex", "dsh"):
        prompt = (ROOT / "prompts" / f"{method}.md").read_text(encoding="utf-8")
        assert "must not" in prompt
        assert "torchport" in prompt.lower()
