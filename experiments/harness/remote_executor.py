"""Trusted remote TorchPort execution for a Harness gateway.

This adapter transports raw CLI reports back to the controller. Feedback filtering,
query budgets and event accounting remain exclusively in HarnessGateway.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from torchport.runtime import write_json


class RemoteExecutor:
    def __init__(
        self, project_root: Path, evidence_dir: Path, candidate_root: Path,
        remote: Mapping[str, Any] | None = None,
    ) -> None:
        remote = remote or {}
        if remote.get("ssh_env_file"):
            os.environ.pop("TP_SSH_USER", None)
            os.environ.pop("TP_SSH_PASSWORD", None)
        for key, env_name in (("ssh_env_file", "TP_SSH_ENV_FILE"),
                              ("ssh_host", "TP_SSH_HOST"), ("ssh_port", "TP_SSH_PORT")):
            if remote.get(key):
                os.environ[env_name] = str(remote[key])
        remote_scripts = project_root / "scripts" / "remote"
        if str(remote_scripts) not in sys.path:
            sys.path.insert(0, str(remote_scripts))
        from ssh_exec import connect  # noqa: PLC0415
        from torchport_remote import bundle, transport  # noqa: PLC0415
        from torchport_remote.config import load_settings  # noqa: PLC0415

        self.bundle = bundle
        self.transport = transport
        self.connect = connect
        settings = load_settings(project_root)
        defaults = dict(settings.target_env_defaults)
        if remote.get("nvcc_path"):
            defaults["nvcc_path"] = str(remote["nvcc_path"])
        if remote.get("home"):
            defaults["HOME"] = str(remote["home"])
        self.settings = replace(
            settings,
            root=PurePosixPath(remote["root"]) if remote.get("root") else settings.root,
            source_python=str(remote.get("source_python") or settings.source_python),
            target_python=str(remote.get("target_python") or settings.target_python),
            python_config=str(remote.get("python_config") or settings.python_config),
            system_libstdcxx=str(remote.get("system_libstdcxx", settings.system_libstdcxx)),
            target_path=str(remote.get("target_path") or settings.target_path),
            target_env_defaults=defaults,
            extra_roots=(*settings.extra_roots, candidate_root.resolve(), evidence_dir.parent.resolve()),
        )
        self.evidence_dir = evidence_dir
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.sequence = max(
            (int(path.name[:4]) for path in self.evidence_dir.glob("[0-9][0-9][0-9][0-9]-*.json")),
            default=0,
        )

    def call(
        self, tool: str, task_path: Path, *, trial: str, manifest: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if tool not in {"run", "evaluate", "verify", "trace"}:
            raise ValueError("不支持的远端执行工具")
        base = self.bundle.remote_base(task_path, self.settings)
        client = self.connect()
        try:
            self.bundle.sync(client, task_path, base, self.settings)
            args = [tool, str(base / "task.json")]
            if tool == "run":
                args.append("target")
            elif tool == "evaluate":
                if manifest is None:
                    raise ValueError("evaluate 缺少 run manifest")
                run_id = str(manifest["run_id"])
                args.append(str(base / "evidence" / "runs" / trial / run_id / "manifest.json"))
            args += ["--root", str(base / "evidence"), "--trial", trial]
            exports = [f"export PYTHONPATH={shlex.quote(str(base / 'src'))}"]
            if self.settings.system_libstdcxx:
                exports.append(f"export LD_PRELOAD={shlex.quote(self.settings.system_libstdcxx)}")
            command = " && ".join([
                f"cd {shlex.quote(str(base))}", *exports,
                f"{shlex.quote(self.settings.source_python)} -m torchport "
                + " ".join(shlex.quote(arg) for arg in args),
            ])
            code, output, stderr = self.transport.run(client, command, timeout=3900)
        finally:
            client.close()
        self.sequence += 1
        evidence_path = self.evidence_dir / f"{self.sequence:04d}-{tool}.json"
        if code not in {0, 3}:
            write_json(evidence_path, {
                "task": str(task_path), "trial": trial, "exit_code": code,
                "stdout": output, "stderr": stderr,
            })
            raise RuntimeError(f"远端 {tool} 执行失败（退出码 {code}）；原始日志保存在控制端")
        try:
            report = json.loads(output)
        except json.JSONDecodeError as exc:
            write_json(evidence_path, {
                "task": str(task_path), "trial": trial, "exit_code": code,
                "stdout": output, "stderr": stderr,
            })
            raise RuntimeError(f"远端 {tool} 未返回有效 JSON；原始日志保存在控制端") from exc
        if not isinstance(report, dict):
            write_json(evidence_path, {
                "task": str(task_path), "trial": trial, "exit_code": code,
                "stdout": output, "stderr": stderr,
            })
            raise RuntimeError("远端报告不是 JSON 对象")
        write_json(evidence_path, {
            "task": str(task_path), "trial": trial, "exit_code": code,
            "stdout": report, "stderr": stderr,
        })
        return report
