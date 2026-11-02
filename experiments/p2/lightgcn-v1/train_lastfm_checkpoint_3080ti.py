"""Run the authors' LightGCN training entry under a frozen 3080 Ti setup."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    code = source / "code"
    if not (code / "main.py").is_file() or args.output.exists():
        raise FileExistsError("source entry missing or output already exists")
    args.output.mkdir(parents=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch  # noqa: PLC0415

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    options = ["--dataset=lastfm", "--model=lgn", "--layer=3", "--recdim=64",
               "--dropout=0", "--lr=0.001", "--decay=0.0001", "--bpr_batch=2048",
               "--epochs=10", "--seed=20261102", "--topks=[20]", "--testbatch=100",
               "--tensorboard=0", "--multicore=0", "--load=0"]
    (args.output / "protocol.json").write_text(json.dumps({
        "entry": str(code / "main.py"), "options": options,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }, indent=2) + "\n", encoding="utf-8")
    os.chdir(code)
    sys.path.insert(0, str(code))
    sys.argv = [str(code / "main.py"), *options]
    runpy.run_path(str(code / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
