"""Export source propagated tables and logits for LightGCN B_l (FP32 torch and FP64 NumPy).

Loads the frozen trained checkpoint into the authors' LightGCN, records `computer()` user and
item tables and the pre-sigmoid logits of the test users. The FP64 reference repeats the same
graph average in NumPy float64 on the source's own normalised sparse graph.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--users", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, checkpoint = args.source.resolve(), args.checkpoint.resolve()
    output, users_path = args.output.resolve(), args.users.resolve()
    if output.exists():
        raise FileExistsError(output)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch  # noqa: PLC0415
    from scipy import sparse  # noqa: PLC0415

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    code = source / "code"
    os.chdir(code)
    sys.path.insert(0, str(code))
    sys.argv = [str(code / "main.py"), "--dataset=lastfm", "--model=lgn", "--layer=3",
                "--recdim=64", "--dropout=0", "--seed=20261102", "--topks=[20]",
                "--testbatch=100", "--tensorboard=0", "--multicore=0"]
    import register  # noqa: E402, PLC0415
    import world  # noqa: E402, PLC0415

    model = register.MODELS[world.model_name](world.config, register.dataset).to(world.device)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    users = np.load(users_path, allow_pickle=False)
    with torch.no_grad():
        user_table, item_table = model.computer()
        logits = user_table[torch.as_tensor(users, device=user_table.device)] @ item_table.T
    graph = model.Graph.coalesce()
    index = graph.indices().cpu().numpy()
    value = graph.values().cpu().numpy().astype(np.float64)
    adjacency = sparse.csr_matrix((value, (index[0], index[1])), shape=tuple(graph.shape))
    embedding = np.concatenate([weights["embedding_user.weight"].numpy(),
                                weights["embedding_item.weight"].numpy()]).astype(np.float64)
    layers = [embedding]
    for _ in range(3):
        layers.append(adjacency @ layers[-1])
    light = np.mean(np.stack(layers, axis=1), axis=1)
    user64, item64 = light[: model.num_users], light[model.num_users:]
    tables = {
        "user_table": user_table.cpu().numpy(), "item_table": item_table.cpu().numpy(),
        "logits": logits.cpu().numpy(),
        "user_table_fp64": user64, "item_table_fp64": item64, "logits_fp64": user64[users] @ item64.T,
    }
    np.savez(output, **tables)
    report = {name: float(np.abs(tables[name].astype(np.float64) - tables[f"{name}_fp64"]).max())
              for name in ("user_table", "item_table", "logits")}
    print(json.dumps({"fp32_vs_fp64_max_abs": report, "graph_nnz": int(value.size)}))


if __name__ == "__main__":
    main()
