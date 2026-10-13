#!/usr/bin/env python3
"""在本地 DSH 中调用远端 GPU 上的同名 TorchPort 命令。

本文件只保留可执行入口和旧私有符号的兼容层；配置、打包、传输与反馈裁剪分别位于
`torchport_remote/`。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ssh_exec import connect  # noqa: E402
from torchport_remote import bundle, feedback, runner, transport  # noqa: E402
from torchport_remote.config import (  # noqa: E402
    REMOTE_ENV_NAMES,
    load_env_file,
    load_settings,
)

__all__ = ["REMOTE_ENV_NAMES", "load_env_file", "main"]

SETTINGS = load_settings(Path(__file__).resolve().parents[2])

REMOTE_ROOT = SETTINGS.root
REMOTE_SOURCE_PYTHON = SETTINGS.source_python
REMOTE_TARGET_PYTHON = SETTINGS.target_python
REMOTE_PYTHON_CONFIG = SETTINGS.python_config
SYSTEM_LIBSTDCXX = SETTINGS.system_libstdcxx
REMOTE_TARGET_PATH = SETTINGS.target_path
REMOTE_TARGET_ENV_DEFAULTS = SETTINGS.target_env_defaults
WEIGHTS_CACHE = SETTINGS.weights_cache
PROJECT_ROOT = SETTINGS.project_root
EXTRA_ROOTS = SETTINGS.extra_roots
FEEDBACK = SETTINGS.feedback

SKIP_PARTS = bundle.SKIP_PARTS
TASK_COMMANDS = runner.TASK_COMMANDS
COMMANDS = runner.COMMANDS
LOCALIZING_COMMANDS = runner.LOCALIZING_COMMANDS


def _inside_workspace(path: str | Path) -> Path:
    return bundle.inside_workspace(path, SETTINGS)


def _remote_base(task_path: Path) -> PurePosixPath:
    return bundle.remote_base(task_path, SETTINGS)


def _report_base(report: str) -> PurePosixPath:
    return bundle.report_base(report, SETTINGS)


def _local_weights(task_path: Path, task: dict[str, Any]) -> tuple[Path, str] | None:
    return bundle.local_weights(task_path, task, SETTINGS)


def _cached_weights_path(local: Path, sha: str) -> PurePosixPath:
    return bundle.cached_weights_path(local, sha, SETTINGS)


def _ensure_weights(client: Any, local: Path, sha: str) -> None:
    transport.ensure_weights(client, local, sha, _cached_weights_path(local, sha))


def _build_bundle(task_path: Path, base: PurePosixPath) -> bytes:
    return bundle.build_bundle(task_path, base, SETTINGS)


def _run(client: Any, command: str, timeout: float = 1900) -> tuple[int, str, str]:
    return transport.run(client, command, timeout)


def _sync(client: Any, task_path: Path, base: PurePosixPath) -> None:
    bundle.sync(client, task_path, base, SETTINGS)


def _remote_arguments(arguments: list[str], base: PurePosixPath) -> list[str]:
    return runner.remote_arguments(arguments, base)


diagnostic_view = feedback.diagnostic_view


def _filter(command: str, out: str) -> str:
    return feedback.filter_output(command, out, FEEDBACK)


def main(arguments: list[str] | None = None) -> int:
    effective = replace(SETTINGS, feedback=FEEDBACK)
    return runner.execute(
        list(arguments or sys.argv[1:]),
        connect=connect,
        settings=effective,
    )


if __name__ == "__main__":
    raise SystemExit(main())
