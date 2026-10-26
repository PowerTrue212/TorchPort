"""Place SimCSE task bundles on overlay and their raw evidence on data storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/remote"))
from ssh_exec import connect  # noqa: E402

OVERLAY = "/root/SimCSE-Benchmark-v1/harness"
DATA = "/root/data-fs/SimCSE-Benchmark-v1/harness-evidence"


def task_paths(controller: Path) -> list[Path]:
    return [controller / "task-smoke.json", controller / "task-stsb.json",
            controller / "trial/delivery/official-task-inference.json"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    args = parser.parse_args()
    controller = args.controller.resolve()
    if not (controller / "mcp-trial.json").is_file():
        raise FileNotFoundError(controller / "mcp-trial.json")
    client = connect()
    try:
        staged = []
        for path in task_paths(controller):
            digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
            base = f"{OVERLAY}/{digest}"
            evidence = f"{DATA}/{digest}"
            link = f"{base}/evidence"
            command = (
                f"mkdir -p {shlex.quote(base)} {shlex.quote(evidence)} && "
                f"if [ -L {shlex.quote(link)} ]; then "
                f"test \"$(readlink {shlex.quote(link)})\" = {shlex.quote(evidence)}; "
                f"elif [ -e {shlex.quote(link)} ]; then exit 4; "
                f"else ln -s {shlex.quote(evidence)} {shlex.quote(link)}; fi"
            )
            _, stdout, stderr = client.exec_command(command)
            code = stdout.channel.recv_exit_status()
            if code:
                raise RuntimeError(f"Evidence staging failed for {path}: "
                                   f"exit {code}; {stderr.read().decode('utf-8', 'replace')}")
            staged.append({"task_path": str(path), "digest": digest,
                           "remote_evidence": evidence})
    finally:
        client.close()
    output = controller / "remote-evidence-staging.json"
    output.write_text(json.dumps(staged, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"count": len(staged), "manifest": str(output)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
