"""Prepare CLIP P2 inputs and neutral weights without inheriting the old V100-only protocol."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CLIP_ROOT = ROOT / "CLIP"


def _sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    config = json.loads((CLIP_ROOT / "experiment.json").read_text("utf-8"))
    source = CLIP_ROOT / "source" / "openai-clip"
    checkpoint = CLIP_ROOT / "shared" / "checkpoints" / config["model"]["checkpoint_filename"]
    if (source / ".git").is_dir():
        head = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
    else:
        head = (source / ".torchport_commit").read_text("utf-8").strip()
    if head != config["model"]["commit"] or _sha(checkpoint) != config["model"]["checkpoint_sha256"]:
        raise RuntimeError("CLIP source or checkpoint identity mismatch")

    sys.path.insert(0, str(source))
    import torch

    import clip

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    seed = int(config["execution"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model, preprocess = clip.load(str(checkpoint), device=args.device, jit=False)
    model = model.float().eval()
    raw = CLIP_ROOT / "shared" / "raw_images"
    paths = [
        raw / f"{index:02d}_{sample['label']}.png"
        for index, sample in enumerate(config["input"]["samples"])
    ]
    images = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in paths]).to(
        args.device, dtype=torch.float32
    )
    tokens = clip.tokenize([sample["text"] for sample in config["input"]["samples"]]).to(
        args.device
    )
    with torch.inference_mode():
        logits, _ = model(images, tokens)

    prepared = CLIP_ROOT / "shared" / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    np.save(prepared / "images.npy", images.cpu().numpy().astype(np.float32), allow_pickle=False)
    np.save(prepared / "tokens.npy", tokens.cpu().numpy().astype(np.int64), allow_pickle=False)
    np.save(prepared / "logits_ref.npy", logits.cpu().numpy().astype(np.float32), allow_pickle=False)
    np.savez(
        prepared / "weights_fp32.npz",
        **{name: value.detach().cpu().numpy() for name, value in model.state_dict().items()},
    )
    files = {
        str(path.relative_to(CLIP_ROOT).as_posix()): {
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(prepared.iterdir())
        if path.is_file()
    }
    report = {
        "schema_version": 1,
        "kind": "torchport.p2-clip-assets",
        "clip_commit": head,
        "checkpoint_sha256": _sha(checkpoint),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "files": files,
    }
    output = CLIP_ROOT / "shared" / "provenance" / "p2-reference.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
