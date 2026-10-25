"""Prepare controller-only files for one CLIP pilot condition.

Writes, outside the candidate workspace:
- mcp-trial.json, task.json      gateway config and TaskSpec (controller account only)
- mcp-token.txt                  bearer token for the HTTP MCP service
- dsh.patch.yml                  DSH overlay: workspace-write + streamable-http MCP
- preflight.json                 probe config, run as the Agent account before start
- audit-rules.json               rules for audit_trial.py scan after the session

It does not start an Agent, create OS accounts, or change ACLs.
"""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CLIP = ROOT / "migrations" / "CLIP"
BENCH = Path(r"D:\科研\CLIP-Benchmark-v1")

# 人工迁移与本项目仓库；私有仓库对匿名访问返回 404，预检把它视为不可达
FORBIDDEN_URLS = [
    "https://github.com/uyzhang/JCLIP",
    "https://raw.githubusercontent.com/uyzhang/JCLIP/main/README.md",
    "https://github.com/PowerTrue212/Jittor-Harness",
    "https://github.com/PowerTrue212/TorchPort",
    "https://github.com/qinxi1477/TorchPort",
]
URL_RULES = ["uyzhang/jclip", "jclip", "powertrue212/jittor-harness",
             "powertrue212/torchport", "qinxi1477/torchport"]
REQUIRED_URLS = ["https://pypi.org/simple/jittor/", "https://github.com/openai/CLIP"]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("none", "full"), required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--controller-dir", type=Path)
    parser.add_argument("--ssh-env-file", type=Path, default=Path.home() / ".tp_ssh.env")
    args = parser.parse_args()

    other = "full" if args.condition == "none" else "none"
    candidate = (BENCH / "workspaces" / args.condition).resolve()
    controller = (args.controller_dir or BENCH / "controller" / args.trial_id).resolve()
    if not (candidate / "evaluation_entry.py").is_file():
        raise ValueError(f"候选工作区或固定入口缺失：{candidate}")
    if controller == candidate or candidate in controller.parents or controller in candidate.parents:
        raise ValueError("控制目录不能位于候选目录内或其父目录")
    if controller.exists():
        raise FileExistsError(f"控制目录已存在，不覆盖：{controller}")
    controller.mkdir(parents=True)

    task = json.loads((CLIP / "outputs" / f"task-forward-{args.condition}.json")
                      .read_text(encoding="utf-8"))
    if Path(task["target"]["repo"]).resolve() != candidate:
        raise ValueError("TaskSpec 的 target.repo 与候选工作区不一致")
    write_json(controller / "task.json", task)

    config = {
        "schema_version": 1,
        "status": "pilot_only",
        "project_root": str(ROOT),
        "candidate_root": str(candidate),
        "tasks": {"inference": str(controller / "task.json")},
        "hidden_tasks": {"inference": str(controller / "task.json")},
        "evidence_root": str(controller / "evidence"),
        "trial_dir": str(controller / "trial"),
        "trial_id": args.trial_id,
        "agent": "DSH",
        "feedback": "localized" if args.condition == "full" else "diagnostic",
        "max_oracle_queries": 10,
        "max_wall_seconds": 10800,
        "candidate_entry": "evaluation_entry.py",
        "remote": {
            "ssh_env_file": str(args.ssh_env_file.resolve()),
            "root": "/root/CLIP-Benchmark-v1/harness",
            "source_python": "/root/data-fs/APT/.venv-source/bin/python",
            "target_python": "/root/data-fs/APT/.venv-jittor/bin/python",
            "python_config": "",
            "system_libstdcxx": "",
            "nvcc_path": "/usr/local/cuda/bin/nvcc",
            "home": "/root/CLIP-Benchmark-v1/home",
            "target_path": "/usr/local/cuda/bin:/root/data-fs/APT/.venv-jittor/bin:"
                           "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        },
    }
    write_json(controller / "mcp-trial.json", config)
    token = secrets.token_urlsafe(32)
    (controller / "mcp-token.txt").write_text(token + "\n", encoding="utf-8")

    def quote(value: object) -> str:
        return json.dumps(str(value), ensure_ascii=False)

    (controller / "dsh.patch.yml").write_text(
        "- id: sandbox-policy\n"
        "  config:\n"
        "    mode: workspace-write\n"
        f"    workspaceRoot: {quote(candidate)}\n"
        "- insert:\n"
        "    - id: mcp-migration-check\n"
        "      name: '@deepseek-ai/dsh-mcp-client'\n"
        "      config:\n"
        "        serverName: migration-check\n"
        "        transport: streamable-http\n"
        f"        url: {quote(f'http://127.0.0.1:{args.port}/mcp')}\n"
        "        headers:\n"
        f"          Authorization: {quote('Bearer ' + token)}\n"
        "        toolCallTimeoutMs: 3900000\n"
        "        failOnStartupError: true\n",
        encoding="utf-8",
    )

    home = Path.home()
    forbidden_paths = [
        str(ROOT), str(CLIP), str(BENCH / "controller"), str(BENCH / "source"),
        str(BENCH / "assets"), str(BENCH / "workspaces" / other),
        str(home / ".tp_ssh.env"), str(home / ".claude"), str(home / ".codex"),
        str(home / ".dsh" / "sessions"),
    ]
    write_json(controller / "preflight.json", {
        "workspace": str(candidate),
        "forbidden_paths": forbidden_paths,
        "forbidden_imports": ["torchport", "experiments.harness"],
        "forbidden_urls": FORBIDDEN_URLS,
        "required_urls": REQUIRED_URLS,
    })
    write_json(controller / "audit-rules.json", {
        "forbidden_paths": forbidden_paths,
        "forbidden_urls": URL_RULES,
        "allowed_paths": [],
        "max_steps": 200,
    })
    print(json.dumps({
        "controller": str(controller), "candidate": str(candidate),
        "feedback": config["feedback"], "mcp_url": f"http://127.0.0.1:{args.port}/mcp",
        "status": "pilot_only",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
