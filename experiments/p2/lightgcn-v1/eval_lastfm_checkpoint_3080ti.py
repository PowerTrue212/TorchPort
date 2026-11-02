"""Evaluate a frozen authors' LightGCN checkpoint on the bundled LastFM test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, checkpoint = args.source.resolve(), args.checkpoint.resolve()
    if not checkpoint.is_file() or args.output.exists():
        raise FileExistsError("checkpoint missing or output already exists")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch  # noqa: PLC0415

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    code = source / "code"
    os.chdir(code)
    sys.path.insert(0, str(code))
    sys.argv = [str(code / "main.py"), "--dataset=lastfm", "--model=lgn", "--layer=3",
                "--recdim=64", "--dropout=0", "--seed=20261102", "--topks=[20]",
                "--testbatch=100", "--tensorboard=0", "--multicore=0"]
    import Procedure  # noqa: E402, PLC0415
    import register  # noqa: E402, PLC0415
    import world  # noqa: E402, PLC0415

    dataset = register.dataset
    model = register.MODELS[world.model_name](world.config, dataset).to(world.device)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    loaded = model.load_state_dict(weights, strict=True)
    metrics = Procedure.Test(dataset, model, 10, None, 0)
    result = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "train_data_sha256": sha256(source / "data" / "lastfm" / "data1.txt"),
              "test_data_sha256": sha256(source / "data" / "lastfm" / "test1.txt"),
              "loaded_parameters": len(weights), "missing": loaded.missing_keys,
              "unexpected": loaded.unexpected_keys, "users": dataset.n_users,
              "items": dataset.m_items, "test_users": len(dataset.testDict),
              "topks": world.topks,
              "metrics": {name: value.tolist() for name, value in metrics.items()},
              "torch": torch.__version__, "cuda": torch.version.cuda}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
