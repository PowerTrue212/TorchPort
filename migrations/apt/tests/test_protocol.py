from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scripts.common import continual_metrics, tree_digest

ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return json.loads((ROOT / "experiment.json").read_text(encoding="utf-8"))


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def test_frozen_assets_match_protocol() -> None:
    value = config()
    assert digest(ROOT / value["source"]["archive"], "sha256") == value["source"][
        "archive_sha256"
    ]
    assert digest(ROOT / value["dataset"]["archive"], "md5") == value["dataset"][
        "archive_md5"
    ]
    assert digest(ROOT / value["dataset"]["archive"], "sha256") == value["dataset"][
        "archive_sha256"
    ]


def test_tree_digest_uses_platform_independent_posix_order(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "README.md").write_bytes(b"readme\n")
    (tmp_path / "configs" / "model.py").write_bytes(b"model\n")

    paths = sorted(
        (path for path in tmp_path.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(tmp_path).as_posix().encode("utf-8"),
    )
    expected = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(tmp_path).as_posix().encode("utf-8")
        expected.update(len(relative).to_bytes(8, "big"))
        expected.update(relative)
        expected.update(path.read_bytes())

    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
        "README.md",
        "configs/model.py",
    ]
    assert tree_digest(tmp_path) == expected.hexdigest()


def test_apt_is_a_full_training_protocol() -> None:
    value = config()
    assert value["training"] == {
        "tasks": 10,
        "classes_per_task": 10,
        "epochs_per_task": 30,
        "batch_size": 64,
        "optimizer": "Adam",
        "learning_rate": 0.004,
        "ema_coefficient": 0.7,
        "prompt_dropout": 0.01,
        "seed": 1,
        "repeat": 1,
        "num_workers": 4,
    }
    assert value["comparison"]["accuracy_matrix_shape"] == [10, 10]
    assert value["comparison"]["prediction_shape"] == [10000, 100]
    assert value["comparison"]["primary_metrics"] == [
        "RN",
        "Average Accuracy",
        "Forgetting",
        "Delta Accuracy",
        "Rounds",
    ]


def test_three_methods_and_torchport_isolation() -> None:
    methods = config()["methods"]
    assert [item["id"] for item in methods] == ["codex", "dsh", "dsh_torchport"]
    assert {item["id"]: item["torchport_allowed"] for item in methods} == {
        "codex": False,
        "dsh": False,
        "dsh_torchport": True,
    }
    for method in ("codex", "dsh", "dsh_torchport"):
        prompt = (ROOT / "prompts" / f"{method}.md").read_text(encoding="utf-8")
        assert "run_evaluate.sh" in prompt
        assert "record_round.py" in prompt
        assert "10 class-incremental tasks" in prompt


def test_clip_style_structure_with_apt_specific_reference_stage() -> None:
    for relative in (
        "run_prepare.sh",
        "run_reference.sh",
        "run_evaluate.sh",
        "run_report.sh",
        "scripts/bootstrap.py",
        "scripts/prepare_reference.py",
        "scripts/init_workspaces.py",
        "scripts/evaluate_jittor.py",
        "scripts/build_report.py",
        "templates/submission.py",
    ):
        assert (ROOT / relative).is_file()
    contract = (ROOT / "templates" / "submission.py").read_text(encoding="utf-8")
    assert "accuracy_matrix.npy" in contract
    assert "predictions.npz" in contract
    report = (ROOT / "scripts" / "build_report.py").read_text(encoding="utf-8")
    assert "accuracy_matrices.png" in report
    assert "method_table.png" in report


def test_average_accuracy_and_forgetting_follow_paper_definition() -> None:
    matrix = np.full((10, 10), np.nan, dtype=np.float32)
    for task in range(10):
        matrix[task, : task + 1] = 90.0
    matrix[-1, :9] = 80.0
    matrix[-1, 9] = 90.0
    metrics = continual_metrics(matrix)
    assert metrics["average_accuracy_percent"] == 81.0
    assert metrics["forgetting_percent"] == 10.0
    assert metrics["seen_accuracy_percent"][0] == 90.0
    assert metrics["seen_accuracy_percent"][-1] == 81.0
