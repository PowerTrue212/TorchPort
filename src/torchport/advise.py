from __future__ import annotations

import json
from typing import Any


def build_advice(
    migration: dict[str, Any],
    check: dict[str, Any],
    certificate: dict[str, Any] | None,
    align: dict[str, Any],
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    probe = migration.get("probe", {})
    _probe(actions, probe)
    _weights(actions, probe)
    _check(actions, check)
    _align(actions, certificate, align)
    return {
        "ready": not actions,
        "summary": _summary(actions, certificate, align),
        "actions": actions,
    }


def _summary(
    actions: list[dict[str, Any]],
    certificate: dict[str, Any] | None,
    align: dict[str, Any],
) -> str:
    if actions:
        return "需要补充信息或处理失败项。"
    if certificate is not None:
        return "已完成迁移、运行检查和数值证书。"
    if not align.get("requested"):
        return "已完成迁移和运行检查，未请求数值对齐。"
    return "已完成迁移和运行检查。"


def _probe(actions: list[dict[str, Any]], probe: dict[str, Any]) -> None:
    status = probe.get("status")
    if status == "ready":
        return
    if not probe.get("entry"):
        actions.append(
            _action(
                "set-entry",
                "migration",
                "指定仓库运行入口。",
                "--entrypoint",
                "main.py",
            )
        )
    if not probe.get("model"):
        candidates = probe.get("candidates") or []
        example = candidates[0]["name"] if candidates else "main.Model"
        actions.append(
            _action(
                "set-model",
                "migration",
                "指定入口可访问的模型类或工厂函数。",
                "--model",
                example,
                candidates=candidates,
            )
        )
    required = probe.get("required") or []
    if status == "needs-args" or required:
        names = [str(item) for item in required]
        example = _arg_example(probe, names)
        hints = {
            key: value
            for key, value in example.items()
            if value != "..."
        }
        actions.append(
            _action(
                "set-args",
                "migration",
                "补齐模型构造参数。",
                "--model-kw",
                json.dumps(example, ensure_ascii=False),
                fields=names,
                hints=hints,
            )
        )
    if status == "needs-input" or not probe.get("inputs"):
        specs = _input_example(probe)
        example = specs[0] if specs else {"shape": [1, 3, 224, 224], "dtype": "float32"}
        actions.append(
            _action(
                "set-input",
                "migration",
                "指定模型输入形状和类型。",
                "--input",
                json.dumps(example, ensure_ascii=False),
                hints=specs,
            )
        )


def _weights(actions: list[dict[str, Any]], probe: dict[str, Any]) -> None:
    candidates = probe.get("weight_candidates") or []
    if candidates:
        actions.append(
            _action(
                "set-weights",
                "align",
                "发现多个检查点，需要选择源端权重。",
                "--weights",
                str(candidates[0]),
                candidates=candidates,
            )
        )


def _arg_example(probe: dict[str, Any], names: list[str]) -> dict[str, Any]:
    if not names:
        return {"width": 4}
    for candidate in _ordered(probe):
        example = candidate.get("example_kw") or candidate.get("hints") or {}
        if any(name in example for name in names):
            return {name: example.get(name, "...") for name in names}
    return {name: "..." for name in names}


def _input_example(probe: dict[str, Any]) -> list[dict[str, Any]]:
    for candidate in _ordered(probe):
        specs = candidate.get("input_specs") or []
        if specs:
            return specs
    return []


def _ordered(probe: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = list(probe.get("candidates") or [])
    model = probe.get("model")
    if not model:
        return candidates
    matches = [
        item
        for item in candidates
        if item.get("name") == model or str(item.get("name", "")).endswith(f".{model}")
    ]
    rest = [item for item in candidates if item not in matches]
    return [*matches, *rest]


def _check(actions: list[dict[str, Any]], check: dict[str, Any]) -> None:
    if check.get("passed", False):
        return
    syntax = check.get("syntax", {})
    if not syntax.get("passed", True):
        actions.append(
            _action(
                "fix-syntax",
                "check",
                "目标端存在 Python 语法错误，先修正转换规则。",
                "errors",
                syntax.get("errors", []),
            )
        )
        return
    residue = check.get("residue", {})
    if not residue.get("passed", True):
        actions.append(
            _action(
                "fix-imports",
                "check",
                "目标端仍有 PyTorch 依赖残留，需要补导入或 API 迁移规则。",
                "imports",
                residue.get("imports", []),
            )
        )
        return
    runtime = check.get("runtime", {})
    if check.get("manifest_blockers") and not any(
        item["stage"] == "migration" for item in actions
    ):
        actions.append(
            _action(
                "fix-blockers",
                "migration",
                "迁移清单仍有阻塞项，先处理 findings 中的 blocker。",
                "findings",
                "blocker",
            )
        )
    elif runtime.get("attempted") and not runtime.get("passed", True):
        actions.append(
            _action(
                "fix-runtime",
                "check",
                "目标端入口运行失败，先查看标准错误并补迁移规则或运行依赖。",
                "stderr",
                runtime.get("stderr", "")[-800:],
            )
        )


def _align(
    actions: list[dict[str, Any]],
    certificate: dict[str, Any] | None,
    align: dict[str, Any],
) -> None:
    if certificate is not None:
        if certificate.get("passed"):
            return
        first = certificate.get("fp32", {}).get("first")
        actions.append(
            _action(
                "fix-parity",
                "align",
                "数值证书未通过，需要定位首次分歧并补兼容规则。",
                "first",
                first,
            )
        )
        return
    error = align.get("error")
    if error:
        actions.append(
            _action(
                "fix-align",
                "align",
                "数值采集失败，需要修正探针、权重或运行依赖。",
                "error",
                error[-800:],
            )
        )


def _action(
    code: str,
    stage: str,
    message: str,
    flag: str,
    example: Any,
    **extra: Any,
) -> dict[str, Any]:
    result = {
        "id": code,
        "stage": stage,
        "message": message,
        "flag": flag,
        "example": example,
    }
    result.update({key: value for key, value in extra.items() if value})
    return result
