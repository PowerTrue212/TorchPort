from __future__ import annotations

import os
import random
import subprocess
import sys
import time

import numpy as np
from common import ROOT, artifact, digest, load_config, write_json


class RAFTArgs(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def main() -> int:
    config = load_config()
    execution = config["execution"]
    expected_commit = config["model"]["commit"]
    actual_commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "source" / "princeton-vl-raft"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_commit != expected_commit:
        raise RuntimeError(f"RAFT source is at {actual_commit}, expected {expected_commit}")

    sys.path.insert(0, str(ROOT / "source" / "princeton-vl-raft" / "core"))
    import scipy
    import torch

    from raft import RAFT

    if torch.__version__ != config["environment"]["pytorch"]:
        raise RuntimeError(
            f"frozen protocol requires torch {config['environment']['pytorch']}, "
            f"got {torch.__version__}"
        )
    if np.__version__ != config["environment"]["numpy"]:
        raise RuntimeError(
            f"frozen protocol requires numpy {config['environment']['numpy']}, got {np.__version__}"
        )
    if scipy.__version__ != config["environment"]["scipy"]:
        raise RuntimeError(
            f"frozen protocol requires scipy {config['environment']['scipy']}, "
            f"got {scipy.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen protocol")
    device_name = torch.cuda.get_device_name(0)
    if device_name != execution["gpu"]:
        raise RuntimeError(f"frozen protocol requires {execution['gpu']}, got {device_name}")

    checkpoint = ROOT / "shared" / "checkpoints" / config["model"]["checkpoint_filename"]
    if checkpoint.stat().st_size != config["model"]["checkpoint_bytes"]:
        raise RuntimeError("checkpoint size mismatch")
    if digest(checkpoint) != config["model"]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA256 mismatch")

    seed = int(execution["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args = RAFTArgs(
        small=bool(execution["small"]),
        mixed_precision=bool(execution["mixed_precision"]),
        alternate_corr=bool(execution["alternate_corr"]),
        dropout=0,
    )
    model = RAFT(args)
    # PyTorch 2.6 weights_only mode prevents arbitrary pickle code execution.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("checkpoint is not a non-empty state dict")
    normalized = {
        (name[7:] if name.startswith("module.") else name): tensor
        for name, tensor in state.items()
    }
    if not all(isinstance(name, str) and isinstance(tensor, torch.Tensor) for name, tensor in normalized.items()):
        raise RuntimeError("checkpoint contains values other than tensors")
    model.load_state_dict(normalized, strict=True)
    model = model.float().cuda().eval()

    # The same framework-neutral state archive is shared by all three methods.
    weights_path = ROOT / "shared" / "prepared" / "weights_fp32.npz"
    np.savez(
        weights_path,
        **{
            name: tensor.detach().cpu().numpy()
            for name, tensor in normalized.items()
        },
    )

    prepared = ROOT / "shared" / "prepared"
    image1_array = np.load(prepared / "image1.npy", allow_pickle=False)
    image2_array = np.load(prepared / "image2.npy", allow_pickle=False)
    expected_input = tuple(config["fixture"]["image_shape"])
    for name, value in (("image1", image1_array), ("image2", image2_array)):
        if value.shape != expected_input or value.dtype != np.float32 or not np.isfinite(value).all():
            raise RuntimeError(f"invalid {name}: {value.shape} {value.dtype}")

    image1 = torch.from_numpy(image1_array).cuda()
    image2 = torch.from_numpy(image2_array).cuda()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        flow_low, flow_up = model(
            image1,
            image2,
            iters=int(execution["iters"]),
            test_mode=bool(execution["test_mode"]),
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated()

    flow_low_array = flow_low.detach().cpu().numpy().astype(np.float32, copy=False)
    flow_up_array = flow_up.detach().cpu().numpy().astype(np.float32, copy=False)
    expected_low = tuple(config["outputs"]["flow_low_shape"])
    expected_up = tuple(config["outputs"]["flow_up_shape"])
    if flow_low_array.shape != expected_low or not np.isfinite(flow_low_array).all():
        raise RuntimeError(f"invalid flow_low: {flow_low_array.shape} {flow_low_array.dtype}")
    if flow_up_array.shape != expected_up or not np.isfinite(flow_up_array).all():
        raise RuntimeError(f"invalid flow_up: {flow_up_array.shape} {flow_up_array.dtype}")

    flow_low_path = prepared / "flow_low_ref.npy"
    flow_up_path = prepared / "flow_up_ref.npy"
    np.save(flow_low_path, np.ascontiguousarray(flow_low_array), allow_pickle=False)
    np.save(flow_up_path, np.ascontiguousarray(flow_up_array), allow_pickle=False)

    artifact_paths = [
        ROOT / "shared" / "checkpoints" / "raft-things.pth",
        prepared / "image1.npy",
        prepared / "image2.npy",
        weights_path,
        flow_low_path,
        flow_up_path,
    ]
    files = {
        str(path.relative_to(ROOT)).replace("\\", "/"): artifact(path)
        for path in artifact_paths
    }
    reference_environment = {
        "device": device_name,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": peak_memory,
        "flow_low_min": float(flow_low_array.min()),
        "flow_low_max": float(flow_low_array.max()),
        "flow_up_min": float(flow_up_array.min()),
        "flow_up_max": float(flow_up_array.max()),
        "checkpoint_loaded_with_weights_only": True,
    }
    write_json(
        ROOT / "shared" / "provenance" / "reference.json",
        {
            "protocol_version": config["protocol_version"],
            "raft_commit": actual_commit,
            "execution": execution,
            "reference_environment": reference_environment,
            "files": files,
        },
    )
    print(
        f"reference prepared on {device_name}: flow_low={flow_low_array.shape}, "
        f"flow_up={flow_up_array.shape}, elapsed={elapsed:.3f}s"
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
