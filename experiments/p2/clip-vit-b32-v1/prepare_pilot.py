"""Prepare a fixed CIFAR-10 pilot sample and score official CLIP on a GPU."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
PROMPTS = tuple(
    f"a photo of {'an' if name[0] in 'aeiou' else 'a'} {name}"
    for name in CLASS_NAMES
)
SAMPLE_PER_CLASS = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_test_batch(archive: Path) -> tuple[np.ndarray, np.ndarray]:
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.getmember("cifar-10-batches-py/test_batch")
        data = pickle.load(io.BytesIO(bundle.extractfile(member).read()), encoding="bytes")
    images = np.asarray(data[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(data[b"labels"], dtype=np.int64)
    if images.shape != (10000, 3, 32, 32) or labels.shape != (10000,):
        raise ValueError("Unexpected CIFAR-10 test batch shape")
    return images, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.checkpoint) != "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af":
        raise ValueError("CLIP checkpoint SHA-256 differs from the frozen source")
    if sha256(args.archive) != "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce":
        raise ValueError("CIFAR-10 archive SHA-256 differs from the frozen source")
    sys.path.insert(0, str(args.source_repo))
    import clip  # noqa: PLC0415

    raw_images, raw_labels = load_test_batch(args.archive)
    indices = np.concatenate([
        np.flatnonzero(raw_labels == label)[:SAMPLE_PER_CLASS]
        for label in range(len(CLASS_NAMES))
    ])
    if indices.size != 100 or np.unique(indices).size != 100:
        raise ValueError("The balanced sample is incomplete")
    labels = raw_labels[indices]

    model, preprocess = clip.load(str(args.checkpoint), device="cuda", jit=False)
    model.float().eval()
    images = np.stack([
        preprocess(Image.fromarray(raw_images[index].transpose(1, 2, 0))).numpy()
        for index in indices
    ]).astype(np.float32)
    tokens = clip.tokenize(PROMPTS).numpy().astype(np.int64)
    image_tensor = torch.from_numpy(images).cuda()
    text_tensor = torch.from_numpy(tokens).cuda()
    results = []
    with torch.no_grad():
        for _ in range(3):
            chunks = []
            for batch in image_tensor.split(10):
                logits, _ = model(batch, text_tensor)
                chunks.append(logits.detach().cpu().numpy())
            results.append(np.concatenate(chunks, axis=0))
    logits = np.stack(results).astype(np.float32)
    predictions = logits.argmax(axis=-1)
    accuracy = (predictions == labels).mean(axis=1)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "images.npy", images)
    np.save(output / "tokens.npy", tokens)
    np.save(output / "labels.npy", labels)
    np.save(output / "indices.npy", indices.astype(np.int64))
    np.save(output / "source_logits.npy", logits)
    manifest = {
        "model": "OpenAI CLIP ViT-B/32",
        "dataset": "CIFAR-10 Python test_batch",
        "sample_rule": "first 10 test examples of each class, in class order",
        "sample_count": int(indices.size),
        "prompts": PROMPTS,
        "checkpoint_sha256": sha256(args.checkpoint),
        "archive_sha256": sha256(args.archive),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "repeats": 3,
        "source_top1_accuracy": accuracy.tolist(),
        "source_prediction_match_rate": float((predictions == predictions[0]).mean()),
        "source_logits_max_run_delta": float(np.max(np.abs(logits - logits[0]))),
        "files": {path.name: sha256(path) for path in output.glob("*.npy")},
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
