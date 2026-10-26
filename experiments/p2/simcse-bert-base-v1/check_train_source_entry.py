"""Check the Harness source entry against the frozen two-step source probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(str(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch  # noqa: PLC0415

    sys.path.insert(0, str(args.source_repo.resolve()))
    from torchport_train import build_optimizer, load_train_model, train_step  # noqa: PLC0415

    torch.manual_seed(20261027)
    torch.cuda.manual_seed_all(20261027)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_train_model()
    initial = args.reference_dir / "train_initial.npz"
    with np.load(initial, allow_pickle=False) as archive:
        state = {name: torch.from_numpy(np.asarray(archive[name]))
                 for name in archive.files}
    model.load_state_dict(state, strict=True)
    model.cuda().train()
    reference = json.loads((args.reference_dir / "report.json").read_text("utf-8"))
    initial_digest = state_sha256(model)
    if initial_digest != reference["initial_state_sha256"]:
        raise ValueError("Source entry loaded a different initial model")
    batch = {name: torch.from_numpy(np.load(args.batch_dir / f"{name}.npy",
                                           allow_pickle=False)).cuda()
             for name in ("input_ids", "attention_mask", "token_type_ids")}
    optimizer = build_optimizer(model)
    steps = []
    for index in range(2):
        loss = train_step(model, optimizer, **batch)
        observed = {
            "index": index,
            "loss": float(loss.detach().cpu()),
            "state_sha256": state_sha256(model),
        }
        expected = reference["step_reports"][index]
        observed["matches_reference"] = (
            observed["loss"] == expected["loss"]
            and observed["state_sha256"] == expected["state_sha256_after_step"]
        )
        steps.append(observed)
    result = {"initial_state_sha256": initial_digest, "steps": steps,
              "passed": all(step["matches_reference"] for step in steps)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
