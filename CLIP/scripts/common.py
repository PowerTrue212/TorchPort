from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiment.json"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def method_config(method: str) -> dict[str, Any]:
    for item in load_config()["methods"]:
        if item["id"] == method:
            return item
    raise ValueError(f"unknown method: {method}")


def method_ids() -> tuple[str, ...]:
    return tuple(item["id"] for item in load_config()["methods"])


def digest(path: Path, algorithm: str = "sha256") -> str:
    checksum = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


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
    lines = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
    return lines


def prepared_manifest() -> dict[str, Any]:
    path = ROOT / "shared" / "provenance" / "reference.json"
    if not path.exists():
        raise FileNotFoundError(f"reference is not prepared: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_prepared_files() -> dict[str, Any]:
    manifest = prepared_manifest()
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = digest(path)
        if actual != expected["sha256"]:
            raise RuntimeError(
                f"prepared file changed: {relative}; expected {expected['sha256']}, got {actual}"
            )
    return manifest
