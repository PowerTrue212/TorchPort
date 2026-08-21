from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from torchport.models import MigrationTarget

from .edit import finding
from .rules import TORCH_DEPS


def req_name(line: str) -> str | None:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", line)
    return match.group(1).lower().replace("_", "-") if match else None


def build_deps(
    source: Path,
    output: Path,
    dep_files: Iterable[str],
    target: MigrationTarget,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    kept: list[str] = []
    removed: list[dict[str, str]] = []
    findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    used: list[str] = []
    for rel_path in dep_files:
        path = source / rel_path
        if path.suffix.lower() != ".txt":
            findings.append(
                finding(
                    code="dependencies.non-requirements-metadata",
                    level="warning",
                    file=rel_path,
                    line=None,
                    symbol=None,
                    message="尚未自动重写该依赖元数据。",
                    fix="使用生成的隔离安装脚本。",
                )
            )
            continue
        used.append(rel_path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line_no, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line or line.startswith(("#", "-r", "--requirement")):
                continue
            package = req_name(line)
            if package in TORCH_DEPS:
                removed.append({"file": rel_path, "requirement": raw})
                if package != "torch":
                    level = "warning" if package == "torchvision" else "blocker"
                    if package == "torchvision":
                        message = "TorchVision 不会安装；已使用的接口由源代码规则判定。"
                        fix = "检查是否已生成 portlib，未覆盖的接口会保留为阻塞项。"
                    else:
                        message = "PyTorch 领域包不会自动安装到目标环境。"
                        fix = "确认 JTorch 接口，或迁移到 Jittor 对应实现。"
                    findings.append(
                        finding(
                            code="dependencies.pytorch-domain-package",
                            level=level,
                            file=rel_path,
                            line=line_no,
                            symbol=package,
                            message=message,
                            fix=fix,
                        )
                    )
                continue
            key = line.lower()
            if key not in seen:
                kept.append(raw)
                seen.add(key)

    runtime = ["# 由 TorchPort 生成，请在隔离环境中安装。"]
    if target is MigrationTarget.JITTOR_NATIVE:
        runtime.extend(["jittor>=1.3.11", "numpy<2"])
    runtime.extend(kept)
    meta = output / ".torchport"
    meta.mkdir(parents=True, exist_ok=True)
    req_path = meta / "requirements.runtime.txt"
    req_path.write_text("\n".join(runtime) + "\n", encoding="utf-8")
    result = {
        "source_files": used,
        "runtime_requirements": str(req_path.relative_to(output)),
        "removed_pytorch_requirements": removed,
        "kept_requirements": kept,
    }
    return result, findings
