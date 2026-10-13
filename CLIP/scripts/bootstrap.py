from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from common import ROOT, digest, load_config, write_json
from PIL import Image


def download(url: str, output: Path, algorithm: str, expected: str) -> None:
    if output.exists() and digest(output, algorithm) == expected:
        print(f"verified existing {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    print(f"downloading {url} -> {output}")
    with urllib.request.urlopen(url) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    actual = digest(temporary, algorithm)
    if actual != expected:
        temporary.unlink()
        raise RuntimeError(f"checksum mismatch for {url}: expected {expected}, got {actual}")
    temporary.replace(output)


def clone_clip(repository: str, commit: str, output: Path) -> None:
    if (output / ".git").is_dir():
        head = subprocess.check_output(
            ["git", "-C", str(output), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != commit:
            raise RuntimeError(f"CLIP source is at {head}, expected {commit}: {output}")
        print(f"verified existing CLIP source at {commit}")
        return
    if output.exists():
        raise RuntimeError(f"refusing to replace non-git path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".clone-part")
    if temporary.exists():
        shutil.rmtree(temporary)
    subprocess.run(["git", "clone", "--no-checkout", repository, str(temporary)], check=True)
    subprocess.run(["git", "-C", str(temporary), "checkout", "--detach", commit], check=True)
    head = subprocess.check_output(
        ["git", "-C", str(temporary), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != commit:
        raise RuntimeError(f"git checked out {head}, expected {commit}")
    temporary.replace(output)


def write_cifar_images(archive: Path, samples: list[dict[str, object]], output: Path) -> None:
    member_name = "cifar-10-batches-py/test_batch"
    with tarfile.open(archive, "r:gz") as package:
        member = package.getmember(member_name)
        stream = package.extractfile(member)
        if stream is None:
            raise RuntimeError(f"missing {member_name} in {archive}")
        batch = pickle.load(stream, encoding="bytes")
    data = np.asarray(batch[b"data"], dtype=np.uint8)
    labels = list(batch[b"labels"])
    output.mkdir(parents=True, exist_ok=True)
    for order, sample in enumerate(samples):
        index = int(sample["cifar10_test_index"])
        expected_class = int(sample["cifar10_class_id"])
        if labels[index] != expected_class:
            raise RuntimeError(
                f"CIFAR-10 label mismatch at test index {index}: "
                f"expected {expected_class}, got {labels[index]}"
            )
        pixels = data[index].reshape(3, 32, 32).transpose(1, 2, 0)
        path = output / f"{order:02d}_{sample['label']}.png"
        Image.fromarray(pixels, mode="RGB").save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and verify frozen CLIP fixtures")
    parser.add_argument(
        "--skip-cifar-archive-cache",
        action="store_true",
        help="remove the downloaded CIFAR archive after extracting the four PNG files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    model = config["model"]
    inputs = config["input"]

    source = ROOT / "source" / "openai-clip"
    checkpoint = ROOT / "shared" / "checkpoints" / model["checkpoint_filename"]
    archive = ROOT / "downloads" / "cifar-10-python.tar.gz"
    raw_images = ROOT / "shared" / "raw_images"

    clone_clip(model["repository"], model["commit"], source)
    download(model["checkpoint_url"], checkpoint, "sha256", model["checkpoint_sha256"])
    download(inputs["dataset_url"], archive, "md5", inputs["dataset_md5"])
    write_cifar_images(archive, inputs["samples"], raw_images)

    files = {
        str(path.relative_to(ROOT)): {"sha256": digest(path)}
        for path in sorted(raw_images.glob("*.png"))
    }
    files[str(checkpoint.relative_to(ROOT))] = {"sha256": digest(checkpoint)}
    write_json(
        ROOT / "shared" / "provenance" / "bootstrap.json",
        {
            "protocol_version": config["protocol_version"],
            "clip_repository": model["repository"],
            "clip_commit": model["commit"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            "cifar10_archive_md5": inputs["dataset_md5"],
            "files": files,
            "python": sys.version,
        },
    )
    if args.skip_cifar_archive_cache:
        archive.unlink()
    print("bootstrap complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
