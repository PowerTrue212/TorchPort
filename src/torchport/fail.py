from __future__ import annotations

from typing import Any


def classify(
    migration: dict[str, Any],
    check: dict[str, Any],
    certificate: dict[str, Any] | None,
    align: dict[str, Any],
) -> dict[str, Any] | None:
    if migration.get("status") != "converted":
        return _migration(migration)
    if not check.get("passed", True):
        return _check(check)
    if align.get("requested") and migration.get("probe", {}).get("status") != "ready":
        return _probe(migration.get("probe", {}))
    if align.get("error"):
        return _align_error(align["error"])
    if certificate is not None and not certificate.get("passed", True):
        return _certificate(certificate)
    return None


def _migration(migration: dict[str, Any]) -> dict[str, Any]:
    probe = migration.get("probe", {})
    if probe.get("status") not in {None, "ready"}:
        return _probe(probe)
    blocker = _first_blocker(migration.get("findings", []))
    if blocker is None:
        return _fail("JITTOR_CONVERT_FAILED", "migration", "迁移未完成。")
    code = str(blocker.get("id", ""))
    if code.startswith("dependencies."):
        return _fail("DEPENDENCY_MISSING", "migration", blocker.get("message"), blocker)
    return _fail("JITTOR_CONVERT_FAILED", "migration", blocker.get("message"), blocker)


def _probe(probe: dict[str, Any]) -> dict[str, Any]:
    status = probe.get("status")
    reason = str(probe.get("reason") or "")
    if any(word in reason for word in ("动态", "运行时", "网络", "unsafe")):
        return _fail("DYNAMIC_VALUE_UNSAFE", "entry", reason)
    if status == "missing" and not probe.get("entry"):
        return _fail("ENTRY_NOT_FOUND", "entry", reason or "没有发现入口。")
    if status == "missing":
        return _fail("ENTRY_NOT_FOUND", "entry", reason or "没有发现模型入口。")
    return _fail("CONFIG_UNRESOLVED", "entry", reason or "模型配置未完全恢复。")


def _check(check: dict[str, Any]) -> dict[str, Any]:
    syntax = check.get("syntax", {})
    if not syntax.get("passed", True):
        return _fail("JITTOR_CONVERT_FAILED", "check", "目标端存在 Python 语法错误。", syntax)
    residue = check.get("residue", {})
    if not residue.get("passed", True):
        return _fail("JITTOR_CONVERT_FAILED", "check", "目标端仍有 PyTorch 导入残留。", residue)
    runtime = check.get("runtime", {})
    if runtime.get("attempted") and not runtime.get("passed", True):
        text = "\n".join([str(runtime.get("stderr", "")), str(runtime.get("stdout", ""))])
        if _import_error(text):
            return _fail("DEPENDENCY_MISSING", "check", "目标端运行依赖缺失。", runtime)
        if _model_error(text):
            return _fail("MODEL_INIT_FAILED", "check", "模型初始化失败。", runtime)
        return _fail("JITTOR_RUN_FAILED", "check", "目标端运行失败。", runtime)
    if check.get("manifest_blockers"):
        return _fail("JITTOR_CONVERT_FAILED", "migration", "迁移清单仍有阻塞项。")
    return _fail("JITTOR_RUN_FAILED", "check", "目标端检查未通过。", check)


def _align_error(error: str) -> dict[str, Any]:
    text = error.lower()
    if any(word in text for word in ("source", "pytorch", "源端")):
        return _fail("TORCH_RUN_FAILED", "align", error)
    if any(word in text for word in ("target", "jittor", "目标端")):
        return _fail("JITTOR_RUN_FAILED", "align", error)
    if _model_error(text):
        return _fail("MODEL_INIT_FAILED", "align", error)
    if any(word in text for word in ("weight", "checkpoint", "safetensor", "权重", "检查点")):
        return _fail("WEIGHT_LOAD_FAILED", "align", error)
    return _fail("PARITY_FAILED", "align", error)


def _certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    weights = certificate.get("weights", {})
    checkpoint = certificate.get("checkpoint", {})
    if weights.get("complete") is False or checkpoint.get("complete") is False:
        return _fail("WEIGHT_LOAD_FAILED", "align", "权重或检查点加载不完整。", certificate)
    return _fail("PARITY_FAILED", "align", "数值对齐未通过。", _first(certificate))


def _first(certificate: dict[str, Any]) -> dict[str, Any]:
    return {
        "fp32": certificate.get("fp32", {}).get("first"),
        "strict": certificate.get("strict", {}).get("first"),
        "missing": certificate.get("fp32", {}).get("missing", []),
    }


def _first_blocker(findings: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in findings:
        if item.get("severity") == "blocker":
            return item
    return None


def _import_error(text: str) -> bool:
    lowered = text.lower()
    return (
        "modulenotfounderror" in lowered
        or "no module named" in lowered
        or "importerror" in lowered
    )


def _model_error(text: str) -> bool:
    lowered = text.lower()
    markers = ("__init__", "required positional", "unexpected keyword", "model init")
    return "typeerror" in lowered and any(marker in lowered for marker in markers)


def _fail(
    code: str,
    stage: str,
    message: Any,
    detail: Any | None = None,
) -> dict[str, Any]:
    result = {
        "code": code,
        "stage": stage,
        "message": str(message or code),
    }
    if detail is not None:
        result["detail"] = detail
    return result
