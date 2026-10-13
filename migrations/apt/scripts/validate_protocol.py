from __future__ import annotations

import argparse
import json

import numpy as np
from common import (
    ROOT,
    digest,
    load_config,
    method_ids,
    read_json_lines,
    tree_digest,
    verify_assets,
    verify_reference_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate frozen APT experiment state")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("assets", "bootstrap", "initialized", "reference", "results"),
    )
    return parser.parse_args()


def validate_bootstrap(config: dict) -> dict:
    source = ROOT / "source" / "apt"
    manifest_path = ROOT / "shared" / "provenance" / "bootstrap.json"
    if not source.is_dir() or not manifest_path.is_file():
        raise RuntimeError("bootstrap is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tree_digest(source) != manifest["source_tree_sha256"]:
        raise RuntimeError("bootstrapped source tree changed")
    checkpoint = ROOT / "shared" / "checkpoints" / config["backbone"]["checkpoint_filename"]
    if checkpoint.stat().st_size != int(config["backbone"]["checkpoint_bytes"]):
        raise RuntimeError("pretrained checkpoint size mismatch")
    if digest(checkpoint, "md5") != config["backbone"]["checkpoint_md5"]:
        raise RuntimeError("pretrained checkpoint MD5 mismatch")
    return manifest


def validate_reference(config: dict) -> dict:
    manifest = verify_reference_files()
    predictions = np.load(
        ROOT / "shared" / "reference" / "predictions.npz", allow_pickle=False
    )
    expected = tuple(config["comparison"]["prediction_shape"])
    if predictions["logits"].shape != expected or predictions["logits"].dtype != np.float32:
        raise RuntimeError("invalid PyTorch reference logits")
    for name in ("targets", "predictions", "task_ids"):
        if predictions[name].shape != (10000,) or predictions[name].dtype != np.int64:
            raise RuntimeError(f"invalid PyTorch reference array: {name}")
    summary = json.loads(
        (ROOT / "shared" / "reference" / "summary.json").read_text(encoding="utf-8")
    )
    if len(summary["seen_accuracy_percent"]) != 10 or summary["test_samples"] != 10000:
        raise RuntimeError("invalid PyTorch reference summary")
    if not 0.0 <= float(summary["final_accuracy_percent"]) <= 100.0:
        raise RuntimeError("invalid PyTorch reference accuracy")
    return manifest


def main() -> int:
    args = parse_args()
    config = load_config()
    verify_assets()
    if args.stage == "assets":
        print("APT frozen assets valid")
        return 0

    bootstrap = validate_bootstrap(config)
    if args.stage == "bootstrap":
        print("APT bootstrap state valid")
        return 0

    for method in method_ids():
        workspace = ROOT / "workspaces" / method
        repo = workspace / "repo"
        manifest = json.loads(
            (workspace / "workspace_manifest.json").read_text(encoding="utf-8")
        )
        if manifest["method"] != method:
            raise RuntimeError(f"invalid method manifest: {method}")
        if manifest["initial_source_tree_sha256"] != bootstrap["source_tree_sha256"]:
            raise RuntimeError(f"source provenance mismatch: {method}")
        if not (repo / "submission.py").is_file():
            raise FileNotFoundError(repo / "submission.py")
    if args.stage == "initialized":
        print("APT workspaces initialized")
        return 0

    validate_reference(config)
    if args.stage == "reference":
        print("APT PyTorch reference valid")
        return 0

    maximum = int(config["execution"]["max_repair_rounds"])
    for method in method_ids():
        rounds = read_json_lines(ROOT / "results" / method / "rounds.jsonl")
        result_path = ROOT / "results" / method / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if len(rounds) > maximum or result["rounds"] != len(rounds):
            raise RuntimeError(f"invalid repair rounds: {method}")
        if result["rn"]:
            for filename in ("accuracy_matrix.npy", "predictions.npz"):
                if not (ROOT / "results" / method / filename).is_file():
                    raise FileNotFoundError(ROOT / "results" / method / filename)
    print("APT method results valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
