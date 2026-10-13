from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np
from common import ROOT, digest, load_config, write_json
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the frozen CLIP inputs, FP32 weights, and PyTorch reference logits"
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    if args.device != config["execution"]["device"]:
        raise RuntimeError(
            f"frozen protocol requires {config['execution']['device']}, got {args.device}"
        )

    source = ROOT / "source" / "openai-clip"
    checkpoint = ROOT / "shared" / "checkpoints" / config["model"]["checkpoint_filename"]
    if not source.is_dir() or not checkpoint.is_file():
        raise RuntimeError("run bootstrap.py first")
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != config["model"]["commit"]:
        raise RuntimeError(f"CLIP source commit mismatch: {head}")
    if digest(checkpoint) != config["model"]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA256 mismatch")

    sys.path.insert(0, str(source))
    import torch
    import torchvision

    import clip  # type: ignore[import-not-found]

    expected_torch = config["environment"]["pytorch"]
    if torch.__version__ != expected_torch:
        raise RuntimeError(
            f"frozen protocol requires PyTorch {expected_torch}, got {torch.__version__}"
        )
    expected_torchvision = config["environment"]["torchvision"]
    if torchvision.__version__ != expected_torchvision:
        raise RuntimeError(
            f"frozen protocol requires torchvision {expected_torchvision}, "
            f"got {torchvision.__version__}"
        )
    expected_numpy = config["environment"]["numpy"]
    if np.__version__ != expected_numpy:
        raise RuntimeError(f"frozen protocol requires NumPy {expected_numpy}, got {np.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen V100 protocol")
    device_name = torch.cuda.get_device_name(0)
    if "V100" not in device_name.upper():
        raise RuntimeError(f"frozen protocol requires an NVIDIA V100, got {device_name}")

    seed = int(config["execution"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model, preprocess = clip.load(str(checkpoint), device=args.device, jit=False)
    model = model.float().eval()
    raw_dir = ROOT / "shared" / "raw_images"
    image_paths = [
        raw_dir / f"{index:02d}_{sample['label']}.png"
        for index, sample in enumerate(config["input"]["samples"])
    ]
    images = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in image_paths]).to(
        args.device, dtype=torch.float32
    )
    texts = [sample["text"] for sample in config["input"]["samples"]]
    tokens = clip.tokenize(texts, truncate=False).to(args.device)

    with torch.inference_mode():
        logits_per_image, _ = model(images, tokens)
    images_array = images.detach().cpu().numpy().astype(np.float32, copy=False)
    tokens_array = tokens.detach().cpu().numpy().astype(np.int64, copy=False)
    logits_array = logits_per_image.detach().cpu().numpy().astype(np.float32, copy=False)

    expected_images = tuple(config["input"]["images_shape"])
    expected_tokens = tuple(config["input"]["tokens_shape"])
    if images_array.shape != expected_images or images_array.dtype != np.float32:
        raise RuntimeError(f"unexpected images: {images_array.shape} {images_array.dtype}")
    if tokens_array.shape != expected_tokens or tokens_array.dtype != np.int64:
        raise RuntimeError(f"unexpected tokens: {tokens_array.shape} {tokens_array.dtype}")
    if logits_array.shape != (4, 4) or logits_array.dtype != np.float32:
        raise RuntimeError(f"unexpected logits: {logits_array.shape} {logits_array.dtype}")
    if not np.isfinite(logits_array).all():
        raise RuntimeError("reference logits contain NaN or infinity")

    prepared = ROOT / "shared" / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    np.save(prepared / "images.npy", images_array, allow_pickle=False)
    np.save(prepared / "tokens.npy", tokens_array, allow_pickle=False)
    np.save(prepared / "logits_ref.npy", logits_array, allow_pickle=False)

    # A single framework-neutral FP32 state archive is shared by all methods. It is
    # derived once from the same official checkpoint and does not contain migrated code.
    state = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    np.savez(prepared / "weights_fp32.npz", **state)

    relative_files = [
        "shared/prepared/images.npy",
        "shared/prepared/tokens.npy",
        "shared/prepared/logits_ref.npy",
        "shared/prepared/weights_fp32.npz",
        f"shared/checkpoints/{config['model']['checkpoint_filename']}",
    ]
    files = {
        relative: {"sha256": digest(ROOT / relative), "bytes": (ROOT / relative).stat().st_size}
        for relative in relative_files
    }
    write_json(
        ROOT / "shared" / "provenance" / "reference.json",
        {
            "protocol_version": config["protocol_version"],
            "clip_commit": head,
            "model": config["model"]["name"],
            "precision": "float32",
            "device": device_name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "numpy_version": np.__version__,
            "texts": texts,
            "image_files": [str(path.relative_to(ROOT)) for path in image_paths],
            "logits_ref": logits_array.tolist(),
            "files": files,
        },
    )
    print(json.dumps({"device": device_name, "logits_ref": logits_array.tolist()}, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
