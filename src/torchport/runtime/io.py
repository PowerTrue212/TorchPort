"""原子 JSON 产物读写。"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def write_json(path: Path, value: dict[str, Any]) -> None:
    """先写临时文件再替换，保证读者看不到半成品。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), "utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

