"""Bind the two calibrated LoRA tasks to one independent candidate workspace.

This prepares controller-only files. It does not start an Agent or claim a formal trial.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASKS = HERE / "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--controller-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=("none", "full"), required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--ssh-env-file", type=Path)
    parser.add_argument("--ssh-host")
    parser.add_argument("--ssh-port", type=int)
    parser.add_argument("--remote-root")
    parser.add_argument("--source-python")
    parser.add_argument("--target-python")
    parser.add_argument("--python-config")
    parser.add_argument("--nvcc-path")
    parser.add_argument("--remote-home")
    args = parser.parse_args()
    remote_requested = any((args.ssh_env_file, args.ssh_host, args.ssh_port, args.remote_root,
                            args.source_python, args.target_python, args.python_config,
                            args.nvcc_path, args.remote_home))
    if remote_requested and not all((args.ssh_env_file, args.ssh_host, args.ssh_port,
                                     args.remote_root, args.source_python, args.target_python,
                                     args.nvcc_path, args.remote_home)):
        raise ValueError("远端配置必须给出 SSH env、主机、端口、根目录、双端 Python、nvcc 和 HOME")

    candidate = args.candidate.resolve()
    controller = args.controller_dir.resolve()
    if not candidate.is_dir() or not (candidate / "evaluation_entry.py").is_file():
        raise ValueError("候选工作区或固定入口缺失")
    if controller == candidate or candidate in controller.parents or controller in candidate.parents:
        raise ValueError("控制文件不能位于候选目录内或其父目录")
    if controller.exists():
        raise FileExistsError(f"控制目录已存在，不覆盖：{controller}")
    controller.mkdir(parents=True)
    task_paths = {}
    for name, source_path, model in (
        ("inference", TASKS / "task-forward.json", "LoRAGPT2SmallEval"),
        ("training", TASKS / "task.json", "load_model"),
    ):
        task = json.loads(source_path.read_text(encoding="utf-8"))
        task["target"].update(repo=str(candidate), entry="evaluation_entry.py", model=model)
        destination = controller / f"{name}-task.json"
        destination.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        task_paths[name] = str(destination)
    config = {
        "schema_version": 1,
        "status": "pilot_only",
        "project_root": str(ROOT),
        "candidate_root": str(candidate),
        "tasks": task_paths,
        "hidden_tasks": task_paths,
        "evidence_root": str(controller / "evidence"),
        "trial_dir": str(controller / "trial"),
        "trial_id": args.trial_id,
        "agent": "DSH",
        "feedback": "localized" if args.condition == "full" else "diagnostic",
        "max_oracle_queries": 10,
        "max_wall_seconds": 10800,
        "candidate_entry": "evaluation_entry.py",
    }
    remote = {
        "ssh_env_file": str(args.ssh_env_file.resolve()) if args.ssh_env_file else None,
        "ssh_host": args.ssh_host,
        "ssh_port": args.ssh_port,
        "root": args.remote_root,
        "source_python": args.source_python,
        "target_python": args.target_python,
        "python_config": args.python_config,
        "nvcc_path": args.nvcc_path,
        "home": args.remote_home,
        "target_path": (
            f"{Path(args.nvcc_path).parent}:{Path(args.target_python).parent}:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            if remote_requested else None
        ),
        "system_libstdcxx": "" if remote_requested else None,
    }
    if remote_requested:
        config["remote"] = {key: value for key, value in remote.items() if value is not None}
    (controller / "mcp-trial.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    def quote(value: Path) -> str:
        return json.dumps(str(value), ensure_ascii=False)
    patch = (
        "- id: sandbox-policy\n"
        "  config:\n"
        "    mode: workspace-write\n"
        f"    workspaceRoot: {quote(candidate)}\n"
        "- insert:\n"
        "    - id: mcp-migration-check\n"
        "      name: '@deepseek-ai/dsh-mcp-client'\n"
        "      config:\n"
        "        serverName: migration-check\n"
        "        transport: stdio\n"
        f"        command: {quote(ROOT / '.venv' / 'Scripts' / 'python.exe')}\n"
        f"        args: {json.dumps([str(ROOT / 'experiments' / 'harness' / 'agents' / 'mcp_server.py'), '--trial', str(controller / 'mcp-trial.json')], ensure_ascii=False)}\n"
        f"        cwd: {quote(candidate)}\n"
        "        toolCallTimeoutMs: 3900000\n"
        "        failOnStartupError: true\n"
    )
    (controller / "dsh.patch.yml").write_text(patch, encoding="utf-8")
    print(json.dumps({"candidate": str(candidate), "controller": str(controller),
                      "status": "pilot_only"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
