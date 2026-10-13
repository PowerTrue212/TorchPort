from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return json.loads((ROOT / "experiment.json").read_text(encoding="utf-8"))


def test_frozen_raft_protocol() -> None:
    value = config()
    assert value["model"]["commit"] == "2888e15a51fa41140771d3f498ed8023cff098d1"
    assert value["model"]["checkpoint_filename"] == "raft-things.pth"
    assert value["fixture"]["image_shape"] == [1, 3, 384, 512]
    assert value["fixture"]["random_generation"] is False
    assert value["execution"] | {} == {
        "device": "cuda",
        "gpu": "Tesla V100-SXM2-16GB",
        "precision": "float32",
        "batch": 1,
        "iters": 12,
        "test_mode": True,
        "small": False,
        "mixed_precision": False,
        "alternate_corr": False,
        "seed": 0,
        "max_repair_rounds": 8,
    }
    assert value["alignment"] == {"rtol": 1e-4, "atol": 1e-4, "equal_nan": False}


def test_method_isolation() -> None:
    value = config()
    assert [item["id"] for item in value["methods"]] == ["codex", "dsh", "dsh_torchport"]
    assert {item["id"]: item["torchport_allowed"] for item in value["methods"]} == {
        "codex": False,
        "dsh": False,
        "dsh_torchport": True,
    }


def test_output_contract() -> None:
    value = config()
    assert value["outputs"]["primary"] == "flow_up"
    assert value["outputs"]["flow_low_shape"] == [1, 2, 48, 64]
    assert value["outputs"]["flow_up_shape"] == [1, 2, 384, 512]
    assert value["outputs"]["channel_order"] == ["dx", "dy"]


def test_cost_metrics_are_secondary_and_nullable() -> None:
    value = config()["cost_metrics"]
    assert value["comparison_role"] == "raw_secondary_data_only"
    assert value["primary_field"] == "agent_elapsed_seconds"
    assert value["missing_value"] is None
    assert "agent_elapsed_kind" in value["fields"]
    assert "cache_read_tokens" in value["fields"]
    assert "evaluator_elapsed_seconds" in value["fields"]


def test_clip_aligned_layout() -> None:
    for name in (
        "outputs",
        "prompts",
        "requirements",
        "results",
        "scripts",
        "shared",
        "source",
        "templates",
        "tests",
        "workspaces",
    ):
        assert (ROOT / name).is_dir()
    assert (ROOT / "source" / "princeton-vl-raft" / "core" / "raft.py").is_file()
    assert (ROOT / "shared" / "checkpoints" / "raft-things.pth").is_file()
    assert (ROOT / "shared" / "prepared" / "flow_up_ref.npy").is_file()
    assert (ROOT / "templates" / "submission.py").is_file()
    for name in ("evaluate_jittor.py", "build_report.py", "init_workspaces.py", "record_round.py"):
        assert (ROOT / "scripts" / name).is_file()
    for method in ("codex", "dsh", "dsh_torchport"):
        assert (ROOT / "workspaces" / method / "repo").is_dir()
        assert (ROOT / "results" / method).is_dir()
        assert (ROOT / "prompts" / f"{method}.md").is_file()


def test_prompts_enforce_method_isolation() -> None:
    for method in ("codex", "dsh"):
        text = (ROOT / "prompts" / f"{method}.md").read_text(encoding="utf-8").lower()
        assert "must not" in text
        assert "torchport" in text
        assert "run_evaluate.sh" in text
        assert "record_round.py" in text
    assisted = (ROOT / "prompts" / "dsh_torchport.md").read_text(encoding="utf-8")
    assert "locate_first_divergence" in assisted
    assert "run_evaluate.sh" in assisted


def test_pure_dsh_has_runtime_tool_isolation() -> None:
    isolation = config()["isolation"]["pure_dsh"]
    assert isolation["launcher"] == "run_dsh_pure.ps1"
    assert isolation["overlay"] == "isolation/dsh-pure.patch.yml"
    assert isolation["required_audit"] == {
        "active_torchport_plugin": False,
        "overlay_disabled": True,
        "passed": True,
    }

    overlay = (ROOT / "isolation" / "dsh-pure.patch.yml").read_text(encoding="utf-8")
    assert "id: torchport" in overlay
    assert "disabled: true" in overlay

    launcher = (ROOT / "scripts" / "launch_dsh_pure.ps1").read_text(encoding="utf-8")
    assert "--dump-config" in launcher
    assert "active_torchport_plugin" in launcher
    assert "TORCHPORT_PYTHONPATH" in launcher

    prompt = (ROOT / "prompts" / "dsh.md").read_text(encoding="utf-8")
    assert "run_dsh_pure.ps1" in prompt
    assert "active_torchport_plugin=false" in prompt


def test_pure_dsh_terminal_failure_is_explicit() -> None:
    result = json.loads((ROOT / "results" / "dsh" / "result.json").read_text(encoding="utf-8"))
    termination = json.loads(
        (ROOT / "results" / "dsh" / "termination.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (ROOT / "results" / "dsh" / "agent_metrics.json").read_text(encoding="utf-8")
    )
    assert result["rn"] is True
    assert result["na"] is False
    assert result["rounds"] == 1
    assert termination["status"] == "stopped_failed"
    assert termination["success"] is False
    assert termination["final_workspace"]["evaluated"] is False
    assert metrics["forbidden_torchport_calls"] == 0
    assert metrics["tool_calls"] > 0
