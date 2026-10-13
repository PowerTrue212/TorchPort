from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
from common import ROOT, load_config, method_ids, read_json_lines, verify_prepared_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate frozen CLIP experiment state")
    parser.add_argument(
        "--stage", required=True, choices=("bootstrap", "prepared", "initialized", "results")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    source = ROOT / "source" / "openai-clip"
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != config["model"]["commit"]:
        raise RuntimeError(f"source commit mismatch: {head}")
    if args.stage == "bootstrap":
        print("bootstrap state valid")
        return 0

    verify_prepared_files()
    prepared = ROOT / "shared" / "prepared"
    arrays = {
        "images": np.load(prepared / "images.npy", allow_pickle=False),
        "tokens": np.load(prepared / "tokens.npy", allow_pickle=False),
        "logits_ref": np.load(prepared / "logits_ref.npy", allow_pickle=False),
    }
    expected = {
        "images": (tuple(config["input"]["images_shape"]), np.dtype("float32")),
        "tokens": (tuple(config["input"]["tokens_shape"]), np.dtype("int64")),
        "logits_ref": ((4, 4), np.dtype("float32")),
    }
    for name, value in arrays.items():
        shape, dtype = expected[name]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise RuntimeError(f"invalid {name}: shape={value.shape} dtype={value.dtype}")
    if args.stage == "prepared":
        print("prepared state valid")
        return 0

    for method in method_ids():
        workspace = ROOT / "workspaces" / method
        if not (workspace / "repo" / "submission.py").is_file():
            raise FileNotFoundError(workspace / "repo" / "submission.py")
        manifest = json.loads((workspace / "workspace_manifest.json").read_text(encoding="utf-8"))
        if manifest["clip_commit"] != head or manifest["method"] != method:
            raise RuntimeError(f"invalid workspace manifest: {method}")
    if args.stage == "initialized":
        print("initialized state valid")
        return 0

    maximum = int(config["execution"]["max_repair_rounds"])
    for method in method_ids():
        rounds = read_json_lines(ROOT / "results" / method / "rounds.jsonl")
        result_path = ROOT / "results" / method / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if len(rounds) > maximum or result["rounds"] != len(rounds):
            raise RuntimeError(f"invalid rounds for {method}")
        if result["rn"] and not (ROOT / "results" / method / "logits.npy").is_file():
            raise FileNotFoundError(ROOT / "results" / method / "logits.npy")
    print("result state valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
