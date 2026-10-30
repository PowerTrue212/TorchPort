"""Run hidden `verify` for each controller negative control, one after another.

Each control gets its own controller directory through `verify_train_reference.py`
(target bound to the control directory, oracle digest checked unchanged). Before each
run the remote data-fs free space is checked; the loop stops below `--min-free-gb`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "remote"))

from ssh_exec import connect  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_FS = "/root/data-fs"


def free_gb() -> float:
    client = connect()
    try:
        _, stdout, _ = client.exec_command(f"df -B1 --output=avail {DATA_FS} | tail -1", timeout=60)
        return int(stdout.read().decode().strip()) / 1024 ** 3
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--trial-prefix", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-home", required=True)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--shared-task", type=Path)
    args = parser.parse_args()
    for name in args.names:
        free = free_gb()
        print(json.dumps({"control": name, "data_fs_free_gb": round(free, 1)}), flush=True)
        if free < args.min_free_gb:
            print(json.dumps({"stopped": "data-fs free space below limit"}), flush=True)
            return
        trial = f"{args.trial_prefix}-{name}"
        result = subprocess.run([
            sys.executable, str(HERE / "verify_train_reference.py"),
            "--task", str(args.task),
            "--candidate-root", str(args.controls / name),
            "--controller", str(args.controller_root / trial),
            "--trial", trial,
            "--remote-root", args.remote_root,
            "--remote-home", args.remote_home,
            *(["--shared-task", str(args.shared_task)] if args.shared_task else []),
        ], capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(json.dumps({"control": name, "exit_code": result.returncode,
                          "stdout": result.stdout.strip()[-800:],
                          "stderr": result.stderr.strip()[-800:]}), flush=True)


if __name__ == "__main__":
    main()
