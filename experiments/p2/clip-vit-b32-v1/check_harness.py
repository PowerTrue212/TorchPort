"""Run the initial seed through the real remote verify path."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.harness.remote_executor import RemoteExecutor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[3]
    os.environ["TP_REMOTE_USE_MKL"] = "0"
    os.environ["TP_REMOTE_USE_CUTT"] = "0"
    executor = RemoteExecutor(
        project, args.evidence.resolve(), args.candidate.resolve(),
        remote={
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
    )
    connect = executor.connect

    def connect_with_keepalive():
        client = connect()
        client.get_transport().set_keepalive(30)
        return client

    executor.connect = connect_with_keepalive
    result = executor.call("verify", args.task.resolve(), trial="seed-smoke")
    print(json.dumps({
        "passed": result.get("passed"),
        "stage": result.get("stage"),
        "error": result.get("error"),
        "report": result.get("report"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
