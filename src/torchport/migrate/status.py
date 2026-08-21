from __future__ import annotations

from typing import Any

from torchport.models import MigrationTarget

from .edit import finding


def probe_issue(
    probe: dict[str, Any],
    entry: str | None,
    target: MigrationTarget,
) -> dict[str, Any] | None:
    status = probe.get("status")
    if status == "ready":
        return None
    level = "blocker" if target is MigrationTarget.JITTOR_NATIVE else "warning"
    details = {
        "needs-args": f"缺少模型构造参数：{', '.join(probe.get('required', []))}",
        "needs-input": "无法推断模型输入形状或类型。",
        "missing": probe.get("reason", "没有发现可验证模型。"),
    }
    return finding(
        code=f"probe.{status or 'missing'}",
        level=level,
        file=entry or ".",
        line=None,
        symbol=probe.get("model"),
        message=details.get(status, "通用数值探针尚未就绪。"),
        fix="使用命令行覆盖模型、构造参数和输入，或补充对应的静态发现规则。",
    )
