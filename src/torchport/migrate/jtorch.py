from __future__ import annotations

import ast
from typing import Any

from .edit import dotted, finding
from .rules import RISK_DEPS


def scan_jtorch(source: str, path: str) -> list[dict[str, Any]]:
    tree = ast.parse(source, filename=path)
    findings: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        name = None
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for module in modules:
                if module in RISK_DEPS or any(
                    module.startswith(f"{item}.") for item in RISK_DEPS
                ):
                    findings.append(
                        finding(
                            code="jtorch.third-party-extension",
                            level="blocker",
                            file=path,
                            line=node.lineno,
                            symbol=module,
                            message="第三方 PyTorch 扩展需要在 JTorch 环境真实验证。",
                            fix="检查目标环境是否加载原生 PyTorch/CUDA 动态库。",
                        )
                    )
        elif isinstance(node, ast.Attribute):
            name = dotted(node)
        if name and name.startswith(("torch.ops.", "torch.classes.", "torch.library.")):
            findings.append(
                finding(
                    code="jtorch.custom-runtime-extension",
                    level="blocker",
                    file=path,
                    line=node.lineno,
                    symbol=name,
                    message="仓库直接调用 PyTorch 原生扩展，JTorch 不能自动证明兼容。",
                    fix="实现 Jittor 算子，或提供纯 Python 回退。",
                )
            )
    return findings
