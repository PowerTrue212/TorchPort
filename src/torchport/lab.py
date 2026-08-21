from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def build_lab(
    paths: Iterable[str | Path],
    names: Iterable[str] = (),
) -> dict[str, Any]:
    rows = _unique([_row(path) for path in _reports(paths)])
    selected = set(names)
    if selected:
        rows = [row for row in rows if row["name"] in selected]
    return {
        "schema_version": 1,
        "kind": "torchport.lab-report",
        "summary": _summary(rows),
        "rows": rows,
    }


def _reports(paths: Iterable[str | Path]) -> list[Path]:
    result: list[Path] = []
    for item in paths:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            result.extend(sorted(path.rglob("auto.json")))
        elif path.is_file():
            result.append(path)
    return result


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(path: Path) -> dict[str, Any]:
    report = _read(path)
    entry = report.get("entry_cert") or {}
    final = entry.get("final") or {}
    failure = report.get("failure") or {}
    agent = report.get("agent") or {}
    return {
        "name": Path(report.get("source") or path.parents[2]).name,
        "report": str(path),
        "mode": agent.get("mode", "auto"),
        "round": agent.get("round", 1),
        "status": report.get("status"),
        "passed": bool(report.get("passed")),
        "entry_status": entry.get("status"),
        "probe_status": entry.get("probe_status"),
        "entry_ref": final.get("ref"),
        "failure": failure.get("code"),
        "stage": failure.get("stage"),
        "agent": agent or None,
    }


def _unique(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("name"), row.get("mode"), row.get("round"))
        if key not in selected or _score(row) > _score(selected[key]):
            selected[key] = row
    return list(selected.values())


def _score(row: dict[str, Any]) -> int:
    return (
        int(row.get("passed")) * 100
        + int(row.get("probe_status") == "ready") * 20
        + int(row.get("entry_status") == "resolved") * 10
        + int(bool(row.get("entry_ref"))) * 5
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        **_metrics(rows),
        "by_mode": _counter(rows, "mode"),
        "by_failure": _counter(rows, "failure"),
        "by_stage": _counter(rows, "stage"),
    }
    result["groups"] = {
        mode: _metrics([item for item in rows if item.get("mode") == mode])
        for mode in sorted({str(item.get("mode")) for item in rows})
    }
    return result


def _metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "passed": _count(rows, "passed", True),
        "with_agent": sum(1 for item in rows if item.get("agent")),
        "entry_found": sum(1 for item in rows if item.get("entry_ref")),
        "entry_resolved": _count(rows, "entry_status", "resolved"),
        "probe_ready": _count(rows, "probe_status", "ready"),
    }


def _count(rows: list[dict[str, Any]], key: str, value: Any) -> int:
    return sum(1 for item in rows if item.get(key) == value)


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    values = Counter(str(item.get(key) or "none") for item in rows)
    return dict(sorted(values.items()))
