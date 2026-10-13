from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
from common import ROOT, artifact, load_config, method_ids, read_json_lines, verify_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the frozen RAFT experiment data")
    parser.add_argument(
        "--stage", choices=("bootstrap", "prepared", "initialized", "results"), required=True
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    source = ROOT / "source" / "princeton-vl-raft"
    actual_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != config["model"]["commit"]:
        raise RuntimeError(f"source commit mismatch: {actual_commit}")
    source_diff = subprocess.run(
        ["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"],
        check=False,
    ).returncode
    if source_diff != 0:
        raise RuntimeError("tracked RAFT source differs from the frozen commit")

    manifest_name = "bootstrap" if args.stage == "bootstrap" else "reference"
    manifest = verify_manifest(manifest_name)
    expected_artifacts = [
        "shared/checkpoints/raft-things.pth",
        "shared/prepared/image1.npy",
        "shared/prepared/image2.npy",
    ]
    if args.stage != "bootstrap":
        expected_artifacts.extend(
            (
                "shared/prepared/flow_low_ref.npy",
                "shared/prepared/flow_up_ref.npy",
                "shared/prepared/weights_fp32.npz",
            )
        )
        if "reference_environment" not in manifest:
            raise RuntimeError("reference environment is missing from provenance")
    for relative in expected_artifacts:
        path = ROOT / relative
        if relative not in manifest["files"]:
            raise RuntimeError(f"artifact is missing from provenance: {relative}")
        if not path.is_file() or artifact(path) != manifest["files"][relative]:
            raise RuntimeError(f"artifact validation failed: {relative}")

    prepared = ROOT / "shared" / "prepared"
    input_shape = tuple(config["fixture"]["image_shape"])
    for name in ("image1", "image2"):
        value = np.load(prepared / f"{name}.npy", allow_pickle=False)
        if value.shape != input_shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise RuntimeError(f"invalid {name}: {value.shape} {value.dtype}")
        if value.min() < 0 or value.max() > 255:
            raise RuntimeError(f"invalid input range for {name}")

    if args.stage != "bootstrap":
        for name in ("flow_low", "flow_up"):
            value = np.load(prepared / f"{name}_ref.npy", allow_pickle=False)
            expected = tuple(config["outputs"][f"{name}_shape"])
            if value.shape != expected or value.dtype != np.float32 or not np.isfinite(value).all():
                raise RuntimeError(f"invalid {name}: {value.shape} {value.dtype}")

    if args.stage in ("bootstrap", "prepared"):
        print(f"RAFT {args.stage} state valid")
        return 0

    for method in method_ids():
        workspace = ROOT / "workspaces" / method
        repo = workspace / "repo"
        if not (repo / "submission.py").is_file():
            raise FileNotFoundError(repo / "submission.py")
        workspace_manifest = json.loads(
            (workspace / "workspace_manifest.json").read_text(encoding="utf-8")
        )
        if workspace_manifest["raft_commit"] != actual_commit:
            raise RuntimeError(f"workspace source mismatch: {method}")
        if workspace_manifest["method"] != method:
            raise RuntimeError(f"workspace method mismatch: {method}")
    if args.stage == "initialized":
        print("RAFT initialized state valid")
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
        if result["rn"] and not (ROOT / "results" / method / "flow_up.npy").is_file():
            raise FileNotFoundError(ROOT / "results" / method / "flow_up.npy")
    print("RAFT results state valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
