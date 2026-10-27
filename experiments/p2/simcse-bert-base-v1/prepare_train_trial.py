"""Write controller-only files for one SimCSE supervised training pilot."""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BENCH = Path("D:/科研/SimCSE-Benchmark-v1")
TASK = BENCH / "controller/prep/task-train-sup-v2.json"
SEED_COMMIT = "340b9713891b723ce8ec6f901049c618b5606b80"
ORACLE_DIGEST = "ca1314b00d197ef04fffa7561b3c28545079847de22dbc6f6666dffe7bb82ab4"

sys.path.insert(0, str(ROOT / "src"))
from torchport.contracts import load_task  # noqa: E402

ADVISORY_URLS = [
    "https://github.com/soMewheRetoFloat/SimCSE-Jittor",
    "https://github.com/PowerTrue212/Jittor-Harness",
    "https://github.com/PowerTrue212/TorchPort",
    "https://github.com/qinxi1477/TorchPort",
]
URL_RULES = [
    "somewheretofloat/simcse-jittor", "simcse-jittor",
    "powertrue212/jittor-harness", "powertrue212/torchport",
    "qinxi1477/torchport",
]


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
    candidate = (BENCH / "workspaces-train" / args.condition).resolve()
    controller = (args.controller_dir or BENCH / "controller" / args.trial_id).resolve()
    if not (candidate / "run.py").is_file():
        raise ValueError(f"Candidate entry is missing: {candidate}")
    commit = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if commit != SEED_COMMIT:
        raise ValueError(f"Candidate is not at {SEED_COMMIT}: {candidate} ({commit})")
    status = subprocess.run(
        ["git", "-C", str(candidate), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout
    if status.strip():
        raise ValueError(f"Candidate checkout is not clean: {candidate}")
    if controller == candidate or candidate in controller.parents or controller in candidate.parents:
        raise ValueError("Controller directory overlaps the candidate checkout")
    if controller.exists():
        raise FileExistsError(f"Controller directory already exists: {controller}")

    task = json.loads(TASK.read_text(encoding="utf-8"))
    task["target"]["repo"] = str(candidate)
    task["target"]["env"]["JITTOR_HOME"] = (
        f"/root/SimCSE-Benchmark-v1/jittor-cache-train-{args.condition}"
    )
    controller.mkdir(parents=True)
    task_path = controller / "task.json"
    write_json(task_path, task)
    original_digest = load_task(TASK)["digest"]
    trial_digest = load_task(task_path)["digest"]
    if original_digest != trial_digest or trial_digest != ORACLE_DIGEST:
        raise ValueError(
            f"Training oracle digest changed: original={original_digest}, "
            f"trial={trial_digest}, expected={ORACLE_DIGEST}"
        )

    config = {
        "schema_version": 1,
        "status": "pilot_only",
        "project_root": str(ROOT),
        "candidate_root": str(candidate),
        "tasks": {"training": str(task_path)},
        "hidden_tasks": {"training": str(task_path)},
        "evidence_root": str(controller / "evidence"),
        "trial_dir": str(controller / "trial"),
        "trial_id": args.trial_id,
        "agent": "DSH",
        "feedback": "localized" if args.condition == "full" else "diagnostic",
        "max_oracle_queries": 10,
        "max_wall_seconds": 10800,
        "candidate_entry": "run.py",
        "remote": {
            "ssh_env_file": str(args.ssh_env_file.resolve()),
            "root": "/root/SimCSE-Benchmark-v1/harness",
            "source_python": "/root/SimCSE-Benchmark-v1/source_python.sh",
            "target_python": "/root/data-fs/APT/.venv-jittor/bin/python",
            "python_config": "",
            "system_libstdcxx": "",
            "nvcc_path": "/usr/local/cuda/bin/nvcc",
            "home": "/root/SimCSE-Benchmark-v1/home",
            "target_path": "/usr/local/cuda/bin:/root/data-fs/APT/.venv-jittor/bin:"
                           "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            # TaskSpec 的源仓库、冻结输入和初始状态都在 BENCH 下；同步打包需显式放行
            "extra_roots": [str(BENCH)],
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
        "    workspaceRoot: \"/work\"\n"
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
    host_paths = [
        str(ROOT), str(BENCH / "controller"), str(BENCH / "source"),
        str(BENCH / "reference"), str(BENCH / "assets"),
        str(BENCH / "workspaces" / other),
        str(BENCH / "workspaces-train" / other), str(home / ".tp_ssh.env"),
        str(home / ".claude"), str(home / ".codex"),
        str(home / ".dsh" / "sessions"),
    ]
    write_json(controller / "preflight.json", {
        "workspace": "/work",
        "forbidden_paths": ["/mnt/c", "/mnt/d", "/mnt/e", "/mnt/wsl", "/root", "/host"],
        "forbidden_imports": ["torchport", "experiments.harness", "jittor_harness"],
        "advisory_urls": ADVISORY_URLS,
        "required_urls": ["https://pypi.org/simple/jittor/",
                          "https://github.com/princeton-nlp/SimCSE"],
        "required_tcp": [f"127.0.0.1:{args.port}"],
    })
    write_json(controller / "audit-rules.json", {
        "forbidden_paths": host_paths + ["/mnt/c/", "/mnt/d/", "/mnt/e/"],
        "forbidden_urls": URL_RULES,
        "allowed_paths": ["/usr/", "/home/node/", "/tmp/", "/proc/", "/etc/"],
        "max_steps": 200,
    })
    write_json(controller / "container-trial.json", {"port": args.port,
                                                       "workspace": "/work"})
    print(json.dumps({"controller": str(controller), "candidate": str(candidate),
                      "feedback": config["feedback"], "status": "pilot_only",
                      "original_digest": original_digest, "trial_digest": trial_digest},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
