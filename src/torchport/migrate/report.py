from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

from torchport.models import MigrationTarget

from .compat import write_lib


def render_runner(
    target: MigrationTarget,
    entry: str | None,
    env: dict[str, str | None],
) -> str:
    setup = ""
    if target is MigrationTarget.JITTOR_NATIVE:
        setup = '''import jittor as jt

device = os.environ.get("TORCHPORT_DEVICE", "cpu")
if device == "cuda":
    jt.flags.use_cuda = 1
elif device != "cpu":
    raise SystemExit(f"尚未配置运行设备：{device}")
'''
    return f'''from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

{setup}
root = Path(__file__).resolve().parents[1]
entry = {entry!r}
contract = {env!r}
if entry is None:
    raise SystemExit("没有识别到入口，请直接运行模块或指定 --entrypoint。")
for name, value in contract.items():
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
paths = [root]
if (root / "src").is_dir():
    paths.insert(0, root / "src")
sys.path[:0] = [str(path) for path in paths]
sys.argv = [str(root / entry), *sys.argv[1:]]
runpy.run_path(str(root / entry), run_name="__main__")
'''


def next_cmds(target: MigrationTarget, entry: str | None) -> list[str]:
    commands = ["bash .torchport/install.sh"]
    if entry and target is MigrationTarget.JTORCH:
        commands.append("python .torchport/run.py")
    elif entry:
        commands.append("TORCHPORT_DEVICE=cpu python .torchport/run.py")
    return commands


def write_files(
    output: Path,
    target: MigrationTarget,
    entry: str | None,
    rules: set[str],
    probe: dict[str, Any],
) -> list[str]:
    meta = output / ".torchport"
    meta.mkdir(parents=True, exist_ok=True)
    install = meta / "install.sh"
    if target is MigrationTarget.JTORCH:
        text = '''#!/usr/bin/env bash
set -euo pipefail
python -m pip install --index-url https://pypi.jittor.org/simple --force-reinstall torch
python -m pip install -r .torchport/requirements.runtime.txt
'''
    else:
        text = '''#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r .torchport/requirements.runtime.txt
'''
    install.write_text(text, encoding="utf-8")
    install.chmod(install.stat().st_mode | stat.S_IXUSR)
    runner = meta / "run.py"
    runner.write_text(
        render_runner(target, entry, probe.get("env", {})),
        encoding="utf-8",
    )
    probe_path = meta / "probe.json"
    probe_path.write_text(
        json.dumps(probe, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    generated = [
        ".torchport/install.sh",
        ".torchport/probe.json",
        ".torchport/run.py",
        ".torchport/requirements.runtime.txt",
        "TORCHPORT_MIGRATION.md",
    ]
    if target is MigrationTarget.JITTOR_NATIVE and any(
        rule.startswith("compat.") for rule in rules
    ):
        generated.append(write_lib(output))
    return generated


def render_readme(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# TorchPort 迁移结果",
        "",
        f"- 状态：`{report['status']}`",
        f"- 目标：`{report['target']}`",
        f"- Python 文件：{summary['python_files']}",
        f"- 自动修改文件：{summary['transformed_files']}",
        f"- 阻塞项：{summary['blockers']}",
        f"- 警告项：{summary['warnings']}",
        "",
        "## 执行",
        "",
        "```bash",
        *report["next_commands"],
        "```",
        "",
        "## 重要说明",
        "",
        "Python 语法通过不等于数值迁移通过。目标端运行后仍需执行数值对齐。",
    ]
    blockers = [item for item in report["findings"] if item["severity"] == "blocker"]
    if blockers:
        lines.extend(["", "## 阻塞项", ""])
        for item in blockers:
            location = item["file"]
            if item["line"]:
                location += f":{item['line']}"
            lines.append(f"- `{location}`：{item['message']} {item['suggestion']}")
    lines.append("")
    return "\n".join(lines)
