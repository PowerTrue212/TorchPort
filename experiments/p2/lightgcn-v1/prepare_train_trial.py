"""Prepare isolated LightGCN five-step training pilot trials on the 3080 Ti."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BENCH = Path("D:/科研/LightGCN-Benchmark-v1")
SOURCE = BENCH / "source/LightGCN-PyTorch-pilot-r2"
TASK = BENCH / "controller/train-ckpt-five-3080ti-r1/task-pilot.json"
ARCHIVE = BENCH / "controller/task-metrics-3080ti-r2.tar.gz"
INPUTS = BENCH / "controller/train-ckpt-five-inputs-r1"
BASELINE = BENCH / "controller/train-ckpt-five-3080ti-r1-calibration-r1/target-repeat-baseline.npz"
WEIGHTS_SHA = "8a42b8bdfd96262faf094126daaba210762515101594813def00f4089956af60"
ORACLE_DIGEST = "cbbc8f7cc516eb7ed633c528bece83316265045248a079f78564482972f63008"
sys.path.insert(0, str(ROOT / "src"))
from torchport.contracts import load_task  # noqa: E402


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def weights_file() -> Path:
    output = BENCH / "controller/train-trial-assets-r1/weights.npz"
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            member = archive.getmember("task-metrics/export-r1/weights.npz")
            with archive.extractfile(member) as source, output.open("xb") as target:
                if source is None:
                    raise ValueError("checkpoint weights missing from evidence archive")
                shutil.copyfileobj(source, target)
    if sha256(output) != WEIGHTS_SHA:
        raise ValueError("checkpoint weight hash differs from frozen evidence")
    return output


def git(*args: str, cwd: Path, env: dict[str, str] | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True, env=env).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("none", "full"), required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--series", default="r1")
    parser.add_argument("--ssh-env-file", type=Path, default=Path.home() / ".tp_ssh2.env")
    args = parser.parse_args()

    if not args.series.isalnum():
        raise ValueError("series must be alphanumeric")
    candidate = BENCH / f"workspaces-train-five-{args.series}" / args.condition
    controller = BENCH / "controller" / args.trial_id
    if candidate.exists() or controller.exists():
        raise FileExistsError(candidate if candidate.exists() else controller)
    if not args.ssh_env_file.is_file():
        raise FileNotFoundError(args.ssh_env_file)
    if sha256(BASELINE) != "d88be0a61fcb54173e23d484f7182fdf05217fb574cd4257a91f3e6b3e657240":
        raise ValueError("frozen repeat baseline changed")
    weights = weights_file()
    candidate.mkdir(parents=True)
    shutil.copy2(BENCH / "workspaces/train-seed-r2/evaluation_entry.py", candidate)
    shutil.copytree(SOURCE / "code", candidate / "upstream/code")
    shutil.copytree(SOURCE / "data/lastfm", candidate / "upstream/data/lastfm")
    shutil.copy2(SOURCE / "README.md", candidate / "upstream/README.md")
    shutil.copy2(SOURCE / "requirements.txt", candidate / "upstream/requirements.txt")
    git("init", "--initial-branch=main", cwd=candidate)
    git("add", ".", cwd=candidate)
    commit_env = {**os.environ, "GIT_AUTHOR_DATE": "2026-11-02T00:00:00Z",
                  "GIT_COMMITTER_DATE": "2026-11-02T00:00:00Z"}
    git("-c", "user.name=TorchPort Pilot", "-c", "user.email=pilot@local.invalid",
        "commit", "-m", "Frozen LightGCN source and empty Jittor entry",
        cwd=candidate, env=commit_env)
    if git("status", "--porcelain", cwd=candidate):
        raise ValueError("candidate seed not clean")

    task = json.loads(TASK.read_text(encoding="utf-8"))
    task["source"]["repo"] = str(SOURCE)
    task["target"]["repo"] = str(candidate)
    # 每个 series 独立的 Jittor 缓存：r1/r2 共用缓存时残留了按错误 python_config 编出的扩展
    task["target"]["env"]["JITTOR_HOME"] = (
        f"/root/autodl-tmp/LightGCN-Benchmark-v1/jittor-cache-trial-{args.series}-{args.condition}"
    )
    for item, name in zip(task["inputs"],
                          ("users.npy", "positive_items.npy", "negative_items.npy"), strict=True):
        item["path"] = str(INPUTS / name)
    task["weights"] = str(weights)
    task["protocol"]["train_noise_baseline"] = str(BASELINE)
    controller.mkdir(parents=True)
    task_path = controller / "task.json"
    write_json(task_path, task)
    actual_digest = load_task(task_path)["digest"]
    if actual_digest != ORACLE_DIGEST:
        raise ValueError(f"source oracle identity changed: {actual_digest}")

    config = {
        "schema_version": 1, "status": "pilot_only", "project_root": str(ROOT),
        "candidate_root": str(candidate), "tasks": {"training": str(task_path)},
        "hidden_tasks": {"training": str(task_path)},
        "evidence_root": str(controller / "evidence"), "trial_dir": str(controller / "trial"),
        "trial_id": args.trial_id, "agent": "DSH",
        "feedback": "localized" if args.condition == "full" else "diagnostic",
        "max_oracle_queries": 10, "max_wall_seconds": 10800,
        "candidate_entry": "evaluation_entry.py",
        "remote": {
            # 主机、端口与密码只从 env 文件读取，不写进仓库
            "ssh_env_file": str(args.ssh_env_file.resolve()),
            "root": f"/root/autodl-tmp/LightGCN-Benchmark-v1/harness-train-five-{args.series}",
            "source_python": "/root/autodl-tmp/envs/source/bin/python",
            "target_python": "/root/autodl-tmp/envs/target/bin/python",
            # target 是 Python 3.9；显式指定，避免回落到系统 miniconda 的 3.12 config
            "python_config": "/root/autodl-tmp/envs/target/bin/python3.9-config",
            "system_libstdcxx": "",
            "nvcc_path": "/usr/local/cuda-12.8/bin/nvcc",
            "home": "/root/autodl-tmp/LightGCN-Benchmark-v1/home",
            "target_path": "/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:"
                           "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "extra_roots": [str(BENCH)],
        },
    }
    write_json(controller / "mcp-trial.json", config)
    token = secrets.token_urlsafe(32)
    (controller / "mcp-token.txt").write_text(token + "\n", encoding="utf-8")
    (controller / "dsh.patch.yml").write_text(
        "- id: sandbox-policy\n  config:\n    mode: workspace-write\n"
        "    workspaceRoot: \"/work\"\n- insert:\n"
        "    - id: mcp-migration-check\n      name: '@deepseek-ai/dsh-mcp-client'\n"
        "      config:\n        serverName: migration-check\n"
        "        transport: streamable-http\n"
        f"        url: \"http://127.0.0.1:{args.port}/mcp\"\n"
        f"        headers:\n          Authorization: \"Bearer {token}\"\n"
        "        toolCallTimeoutMs: 3900000\n        failOnStartupError: true\n",
        encoding="utf-8",
    )
    other = "full" if args.condition == "none" else "none"
    forbidden = [str(ROOT), str(BENCH / "controller"), str(BENCH / "source"),
                 str(BENCH / f"workspaces-train-five-{args.series}" / other),
                 str(Path.home() / ".tp_ssh2.env"), str(Path.home() / ".codex")]
    blocked_urls = ["powertrue212/jittor-harness", "powertrue212/torchport",
                    "qinxi1477/torchport"]
    write_json(controller / "preflight.json", {
        "workspace": "/work", "forbidden_paths": ["/mnt/c", "/mnt/d", "/root", "/host"],
        "forbidden_imports": ["torchport", "experiments.harness", "jittor_harness"],
        "advisory_urls": ["https://github.com/PowerTrue212/TorchPort"],
        "required_urls": ["https://pypi.org/simple/jittor/"],
        "required_tcp": [f"127.0.0.1:{args.port}"],
    })
    write_json(controller / "audit-rules.json", {
        "forbidden_paths": forbidden + ["/mnt/c/", "/mnt/d/"],
        "forbidden_urls": blocked_urls,
        "allowed_paths": ["/usr/", "/home/node/", "/tmp/", "/proc/", "/etc/"],
        "max_steps": 200,
    })
    write_json(controller / "container-trial.json", {"port": args.port, "workspace": "/work"})
    print(json.dumps({"controller": str(controller), "candidate": str(candidate),
                      "seed_commit": git("rev-parse", "HEAD", cwd=candidate),
                      "oracle_digest": actual_digest, "feedback": config["feedback"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
