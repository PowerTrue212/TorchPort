from __future__ import annotations

from typing import Any


def build_cert(
    source: str,
    target: str,
    spec: dict[str, Any],
    source_info: dict[str, Any],
    target_info: dict[str, Any],
    fp32: dict[str, Any],
    strict: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = source_info.get("checkpoint", {"requested": False, "complete": True})
    input_equal = source_info["hashes"]["input"] == target_info["hashes"]["input"]
    weights_complete = source_info["weights"] == target_info["loaded"]
    passed = bool(fp32["passed"] and input_equal and weights_complete and checkpoint["complete"])
    return {
        "schema_version": 1,
        "kind": "torchport.repository-certificate",
        "status": "verified" if passed else "failed",
        "passed": passed,
        "source": source,
        "target": target,
        "spec": spec,
        "frameworks": {
            "source": source_info["framework"],
            "target": target_info["framework"],
        },
        "checkpoint": checkpoint,
        "weights": {
            "source": source_info["weights"],
            "loaded": target_info["loaded"],
            "complete": weights_complete,
            "sha256": source_info["hashes"]["weights"],
        },
        "tensors": {
            "source": source_info["tensors"],
            "target": target_info["tensors"],
            "input_identical": input_equal,
            "source_sha256": source_info["hashes"]["archive"],
            "target_sha256": target_info["hashes"]["archive"],
        },
        "fp32": {
            "passed": fp32["passed"],
            "first": fp32["first_gating_divergence"],
            "first_any": fp32["first_divergence"],
            "missing": fp32["missing_in_candidate"],
            "extra": fp32["extra_in_candidate"],
        },
        "strict": {
            "passed": strict["passed"],
            "first": strict["first_gating_divergence"],
            "first_any": strict["first_divergence"],
            "missing": strict["missing_in_candidate"],
            "extra": strict["extra_in_candidate"],
        },
    }
