from __future__ import annotations

import subprocess

import numpy as np
from common import ROOT, artifact, digest, load_config, write_json
from PIL import Image


def main() -> int:
    config = load_config()
    expected_commit = config["model"]["commit"]
    actual_commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "source" / "princeton-vl-raft"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_commit != expected_commit:
        raise RuntimeError(f"RAFT source is at {actual_commit}, expected {expected_commit}")

    crop = tuple(int(value) for value in config["fixture"]["crop_xyxy"])
    arrays: list[np.ndarray] = []
    for frame in config["fixture"]["source_frames"]:
        path = ROOT / frame["path"]
        actual_hash = digest(path)
        if actual_hash != frame["sha256"]:
            raise RuntimeError(f"source frame checksum mismatch: {path}")
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB").crop(crop), dtype=np.uint8)
        if pixels.shape != (384, 512, 3):
            raise RuntimeError(f"unexpected crop shape for {path}: {pixels.shape}")
        array = pixels.transpose(2, 0, 1)[None].astype(np.float32, copy=False)
        arrays.append(np.ascontiguousarray(array))

    expected_shape = tuple(config["fixture"]["image_shape"])
    for index, array in enumerate(arrays, 1):
        if array.shape != expected_shape or array.dtype != np.float32:
            raise RuntimeError(f"unexpected image{index}: {array.shape} {array.dtype}")
        if array.min() < 0 or array.max() > 255:
            raise RuntimeError(f"image{index} is outside RAFT's 0..255 input range")

    fixture_dir = ROOT / "shared" / "prepared"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    paths = [fixture_dir / "image1.npy", fixture_dir / "image2.npy"]
    for path, array in zip(paths, arrays):
        np.save(path, array, allow_pickle=False)

    checkpoint = ROOT / "shared" / "checkpoints" / config["model"]["checkpoint_filename"]
    if artifact(checkpoint) != {
        "bytes": config["model"]["checkpoint_bytes"],
        "sha256": config["model"]["checkpoint_sha256"],
    }:
        raise RuntimeError("raft-things.pth checksum or size mismatch")

    files = {
        "shared/checkpoints/raft-things.pth": artifact(checkpoint),
        "shared/prepared/image1.npy": artifact(paths[0]),
        "shared/prepared/image2.npy": artifact(paths[1]),
    }
    write_json(
        ROOT / "shared" / "provenance" / "bootstrap.json",
        {
            "protocol_version": config["protocol_version"],
            "raft_commit": actual_commit,
            "crop_xyxy": list(crop),
            "source_frames": config["fixture"]["source_frames"],
            "files": files,
        },
    )
    print("fixture prepared: image1/image2 are float32 [1,3,384,512]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
