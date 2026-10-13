"""远端 TorchPort 命令编排；连接建立由入口脚本注入。"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import PurePosixPath
from typing import Any, Callable

from . import bundle, transport
from .config import RemoteSettings
from .feedback import filter_output

TASK_COMMANDS = {"inspect", "run", "evaluate", "trace", "verify"}
COMMANDS = TASK_COMMANDS | {"localize"}
LOCALIZING_COMMANDS = {"trace", "localize"}


def remote_arguments(arguments: list[str], base: PurePosixPath) -> list[str]:
    values = list(arguments)
    if values[0] in TASK_COMMANDS:
        values[1] = str(base / "task.json")
        if "--root" in values:
            index = values.index("--root")
            values[index + 1] = str(base / "evidence")
    return values


def execute(
    arguments: list[str],
    *,
    connect: Callable[[], Any],
    settings: RemoteSettings,
) -> int:
    if not arguments or arguments[0] not in COMMANDS:
        raise SystemExit("仅支持 inspect/run/evaluate/trace/localize/verify")
    if settings.feedback not in {"diagnostic", "localized"}:
        raise SystemExit(f"未知反馈条件：{settings.feedback}")
    mode = arguments[arguments.index("--mode") + 1] if "--mode" in arguments[:-1] else "normal"
    localizing = arguments[0] in LOCALIZING_COMMANDS or (arguments[0] == "run" and mode != "normal")
    if settings.feedback == "diagnostic" and localizing:
        sys.stdout.write(json.dumps({"error": f"{arguments[0]} 在本实验条件下不可用"}, ensure_ascii=False))
        return 2
    task_path = bundle.inside_workspace(arguments[1], settings) if arguments[0] in TASK_COMMANDS else None
    base = bundle.remote_base(task_path, settings) if task_path else bundle.report_base(arguments[1], settings)
    client = connect()
    try:
        if task_path:
            bundle.sync(client, task_path, base, settings)
        remote_args = remote_arguments(arguments, base)
        quoted = " ".join(shlex.quote(value) for value in remote_args)
        exports = [f"export PYTHONPATH={shlex.quote(str(base / 'src'))}"]
        if settings.system_libstdcxx:
            exports.append(f"export LD_PRELOAD={shlex.quote(settings.system_libstdcxx)}")
        command = " && ".join([
            f"cd {shlex.quote(str(base))}", *exports,
            f"{settings.source_python} -m torchport {quoted}",
        ])
        code, out, err = transport.run(client, command)
        sys.stdout.write(filter_output(arguments[0], out, settings.feedback))
        sys.stderr.write(err)
        return code
    finally:
        client.close()
