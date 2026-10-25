"""Check that the TorchPort source entry matches official CLIP inference."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-entry", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("source_entry", args.source_entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.TorchPortCLIP().cuda().eval()
    with np.load(args.weights, allow_pickle=False) as archive:
        state = {
            f"model.{name}": torch.from_numpy(np.asarray(archive[name]))
            for name in archive.files
        }
    result = model.load_state_dict(state, strict=False)
    images = torch.from_numpy(np.load(args.pilot / "images.npy")).cuda()
    tokens = torch.from_numpy(np.load(args.pilot / "tokens.npy")).cuda()
    reference = np.load(args.pilot / "source_logits.npy")[0]
    with torch.no_grad():
        observed = torch.cat([model(batch, tokens) for batch in images.split(10)]).cpu().numpy()
    delta = float(np.max(np.abs(observed - reference)))
    print({
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "max_abs": delta,
        "predictions_equal": bool(np.array_equal(observed.argmax(-1), reference.argmax(-1))),
    })
    if result.unexpected_keys or delta > 1e-5 or not np.array_equal(observed.argmax(-1), reference.argmax(-1)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
