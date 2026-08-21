from __future__ import annotations

from pathlib import Path
from typing import Any

from .probe.paths import module_name


def build_entry(
    source: str | Path,
    selected: str | None,
    probe: dict[str, Any],
    inspection: dict[str, Any],
) -> dict[str, Any]:
    entry = probe.get("entry")
    model = probe.get("model")
    status = probe.get("status")
    final = _final(source, entry, model)
    chain = _chain(selected, probe, inspection, final)
    return {
        "schema_version": 1,
        "kind": "torchport.entry-certificate",
        "status": "resolved" if status == "ready" else "unresolved",
        "probe_status": status,
        "confidence": "certain" if status == "ready" else "needs-human",
        "reason": probe.get("reason"),
        "chain": chain,
        "final": final,
        "candidates": probe.get("candidates", []),
    }


def _final(
    source: str | Path,
    entry: str | None,
    model: str | None,
) -> dict[str, Any]:
    module = None
    if entry:
        try:
            module = module_name(Path(source) / entry, Path(source))
        except ValueError:
            module = None
    ref = _ref(module, model)
    return {
        "entry": entry,
        "module": module,
        "model": model,
        "ref": ref,
    }


def _chain(
    selected: str | None,
    probe: dict[str, Any],
    inspection: dict[str, Any],
    final: dict[str, Any],
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    entries = list(inspection.get("entrypoints") or [])
    if selected:
        steps.append(
            {
                "step": "entrypoint",
                "value": selected,
                "source": "argument-or-scan",
                "candidates": entries,
            }
        )
    elif entries:
        steps.append({"step": "entrypoint", "value": None, "candidates": entries})
    if probe.get("entry") and probe.get("entry") != selected:
        steps.append(
            {
                "step": "probe-entry",
                "value": probe["entry"],
                "source": "model-definition",
            }
        )
    if probe.get("env"):
        steps.append({"step": "env", "value": probe["env"], "source": "static"})
    if final.get("ref"):
        steps.append({"step": "model", "value": final["ref"], "source": "static"})
    if probe.get("args") or probe.get("kwargs"):
        steps.append(
            {
                "step": "init",
                "args": probe.get("args", []),
                "kwargs": probe.get("kwargs", {}),
                "source": "static",
            }
        )
    if probe.get("inputs"):
        steps.append({"step": "inputs", "value": probe["inputs"], "source": "static"})
    if probe.get("weights"):
        steps.append({"step": "weights", "value": probe["weights"], "source": "scan"})
    return steps


def _ref(module: str | None, model: str | None) -> str | None:
    if not model:
        return None
    if "." in model:
        prefix, _, name = model.rpartition(".")
        return f"{prefix}:{name}" if prefix and name else model
    return f"{module}:{model}" if module else model
