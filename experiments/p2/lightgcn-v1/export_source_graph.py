"""Export the official normalized LastFM graph for a target reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    sys.path.insert(0, str(source / "code"))
    sys.argv = [sys.argv[0], "--dataset=lastfm", "--layer=3", "--recdim=64", "--dropout=0"]
    import dataloader  # noqa: PLC0415

    dataset = dataloader.LastFM(path=str(source / "data" / "lastfm"))
    graph = dataset.getSparseGraph().coalesce()
    np.savez_compressed(
        args.output,
        indices=graph.indices().cpu().numpy(),
        values=graph.values().cpu().numpy(),
        shape=np.asarray(graph.shape, dtype=np.int64),
    )
    print({"shape": list(graph.shape), "nnz": int(graph._nnz()), "output": str(args.output)})


if __name__ == "__main__":
    main()
