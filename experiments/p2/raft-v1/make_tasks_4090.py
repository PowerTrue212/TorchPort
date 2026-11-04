"""Write distinct inference and training RAFT pilot TaskSpecs."""
import json
from pathlib import Path

BENCH = Path("D:/科研/RAFT-Benchmark-v1")
SOURCE = BENCH / "source/RAFT-2888e15a51fa41140771d3f498ed8023cff098d1"
TARGET = BENCH / "controller/target-reference-r1"
INPUTS = BENCH / "controller/inputs-r1"
TARGET.mkdir(parents=True, exist_ok=False)
(TARGET / "evaluation_entry.py").write_text(
    '"""Controller reference pending implementation."""\n'
    'def load_inference_model():\n    raise NotImplementedError\n'
    'def load_train_model():\n    raise NotImplementedError\n', encoding="utf-8")

side_env = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TP_RAFT_DEPS": "/root/autodl-tmp/RAFT-Benchmark-v1/deps"}
target_env = {
    "JITTOR_HOME": "/root/autodl-tmp/RAFT-Benchmark-v1/jittor-cache-control-r1",
    "nvcc_path": "/usr/local/cuda-12.8/bin/nvcc",
    "conv_opt": "1", "use_mkl": "0", "use_cutt": "0",
    "python_config": "/root/autodl-tmp/envs/target/bin/python3.9-config",
    "PATH": "/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
}


def task(name, source_model, target_model, names, profile, train=False):
    directory = BENCH / "controller" / name
    directory.mkdir(parents=True, exist_ok=False)
    items = []
    for item in names:
        path = INPUTS / f"{item}.npy"
        import numpy as np
        array = np.load(path, mmap_mode="r")
        items.append({"shape": list(array.shape), "dtype": str(array.dtype), "path": str(path)})
    protocol = {"profile": profile, "repeats": 2, "timeout": 1200, "pass_k": 8.0}
    if train:
        protocol["train"] = {"optimizer": "build_optimizer", "step": "train_step", "sketch_dim": 16384}
    value = {
        "schema_version": 1, "task_id": f"raft-{name}-4090-pilot-r1",
        "source": {"repo": str(SOURCE), "entry": "source_adapter.py", "model": source_model,
                   "device": "cuda", "python": "/root/autodl-tmp/envs/source/bin/python", "env": side_env},
        "target": {"repo": str(TARGET), "entry": "evaluation_entry.py", "model": target_model,
                   "device": "cuda", "python": "/root/autodl-tmp/envs/target/bin/python", "env": target_env},
        "inputs": items, "weights": str(INPUTS / "weights-fp32.npz"),
        "seed": 20261103, "protocol": protocol,
    }
    (directory / "task.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(directory / "task.json")


task("inference-r1", "load_inference_model", "load_inference_model",
     ["inference-image1", "inference-image2"], "fp32-inference")
task("train-three-r1", "load_train_model", "load_train_model",
     ["train-image1", "train-image2", "train-flow", "train-valid"], "train-step", True)
