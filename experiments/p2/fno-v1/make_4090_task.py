"""Write the frozen five-step FNO pilot TaskSpec before target implementation."""
import json
from pathlib import Path

bench = Path("D:/科研/FNO-Benchmark-v1")
control = bench / "controller/train-five-4090-r1"
inputs = bench / "controller/train-inputs-4090-r2"
target = bench / "controller/target-reference-4090-r1"
if control.exists() or target.exists():
    raise FileExistsError("preserving existing controller task or target")
control.mkdir(parents=True)
target.mkdir(parents=True)
(target / "evaluation_entry.py").write_text(
    '"""Controller-only pending target entry; never ship to an Agent."""\n'
    'def load_train_model():\n'
    '    raise RuntimeError("Jittor FNO reference not implemented")\n',
    encoding="utf-8",
)
task = {
    "schema_version": 1,
    "task_id": "fno-darcy16-train-five-step-4090-pilot-r1",
    "source": {
        "repo": str(bench / "source"),
        "entry": "torchport_train_adapter.py",
        "model": "load_train_model",
        "device": "cuda",
        "python": "/root/autodl-tmp/envs/source/bin/python",
        "env": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TP_FNO_DEPS": "/root/autodl-tmp/FNO-Benchmark-v1/deps-requests:/root/autodl-tmp/FNO-Benchmark-v1/deps-pinned",
        },
    },
    "target": {
        "repo": str(target),
        "entry": "evaluation_entry.py",
        "model": "load_train_model",
        "device": "cuda",
        "python": "/root/autodl-tmp/envs/target/bin/python",
        "env": {
            "JITTOR_HOME": "/root/autodl-tmp/FNO-Benchmark-v1/jittor-cache-control-r1",
            "nvcc_path": "/usr/local/cuda-12.8/bin/nvcc",
            "conv_opt": "1",
            "PATH": "/usr/local/cuda-12.8/bin:/root/autodl-tmp/envs/target/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        },
    },
    "inputs": [
        {"shape": [5, 2, 1, 16, 16], "dtype": "float32", "path": str(inputs / f"{name}.npy")}
        for name in ("x", "y")
    ],
    "weights": str(inputs / "initial-weights.npz"),
    "seed": 20261030,
    "protocol": {
        "profile": "train-step", "repeats": 2, "timeout": 1200,
        "pass_k": 8.0,
        "train": {"optimizer": "build_optimizer", "step": "train_step"},
    },
}
(control / "task.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
print(control / "task.json")
