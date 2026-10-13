from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiment.json"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def method_ids() -> tuple[str, ...]:
    return tuple(item["id"] for item in load_config()["methods"])


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tree_digest(root: Path) -> str:
    value = hashlib.sha256()
    files = (item for item in root.rglob("*") if item.is_file())
    for path in sorted(
        files,
        key=lambda item: item.relative_to(root).as_posix().encode("utf-8"),
    ):
        if "__pycache__" in path.parts or path.name in {".DS_Store"} or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        value.update(len(relative).to_bytes(8, "big"))
        value.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{number}: {error}") from error
    return values


def verify_assets() -> dict[str, Any]:
    config = load_config()
    source = ROOT / config["source"]["archive"]
    dataset = ROOT / config["dataset"]["archive"]
    if digest(source) != config["source"]["archive_sha256"]:
        raise RuntimeError(f"source archive checksum mismatch: {source}")
    if digest(dataset, "md5") != config["dataset"]["archive_md5"]:
        raise RuntimeError(f"dataset archive MD5 mismatch: {dataset}")
    if digest(dataset) != config["dataset"]["archive_sha256"]:
        raise RuntimeError(f"dataset archive SHA256 mismatch: {dataset}")
    return {"source": source, "dataset": dataset}


def reference_manifest() -> dict[str, Any]:
    path = ROOT / "shared" / "provenance" / "reference.json"
    if not path.is_file():
        raise FileNotFoundError(f"PyTorch reference is not complete: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_reference_files() -> dict[str, Any]:
    manifest = reference_manifest()
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = digest(path)
        if actual != expected["sha256"]:
            raise RuntimeError(
                f"reference file changed: {relative}; expected {expected['sha256']}, got {actual}"
            )
    return manifest


def continual_metrics(matrix: np.ndarray) -> dict[str, Any]:
    """Compute the paper's metrics from R[t, i] in percentage units."""
    if matrix.shape != (10, 10):
        raise ValueError(f"expected a [10,10] accuracy matrix, got {matrix.shape}")
    for task in range(10):
        if not np.isfinite(matrix[task, : task + 1]).all():
            raise ValueError(f"missing evaluated accuracy after task {task + 1}")
    average_accuracy = float(np.mean(matrix[-1]))
    forgetting = float(np.mean(np.diag(matrix)[:-1] - matrix[-1, :-1]))
    seen_curve = [float(np.mean(matrix[index, : index + 1])) for index in range(10)]
    return {
        "average_accuracy_percent": average_accuracy,
        "forgetting_percent": forgetting,
        "seen_accuracy_percent": seen_curve,
    }
