from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行原生 Jittor ResNet-18 并采集分层证据")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("artifacts/models/cnn-resnet18/model.safetensors"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/jittor-resnet18/evidence.json"),
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        default=Path("artifacts/runs/jittor-resnet18/cnn-resnet18.npz"),
    )
    parser.add_argument(
        "--input-evidence",
        type=Path,
        help="可选的源框架 NPZ 证据；提供时读取其中的 input，保证逐位相同输入",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def array_summary(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value, dtype=np.float32)
    statistics = array.astype(np.float64, copy=False)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(statistics.mean()),
        "std": float(statistics.std()),
        "min": float(statistics.min()),
        "max": float(statistics.max()),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def main() -> int:
    args = parse_args()
    if args.device == "cpu":
        # Jittor 在导入时探测 CUDA，因此必须提前固定执行后端。
        os.environ["nvcc_path"] = ""

    import numpy as np
    import jittor as jt
    from jittor.models import resnet18
    from safetensors.numpy import load_file

    if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
        raise SystemExit(
            "Jittor 1.3.11.0 需要项目隔离的 NumPy 1.x；"
            "请先运行 scripts/install_jittor_compat_deps.sh，"
            "并把 artifacts/jittor-deps 放在 PYTHONPATH 首位。"
        )

    if args.device == "cuda":
        jt.flags.use_cuda = 1

    weights_path = args.weights.expanduser().resolve()
    if not weights_path.is_file():
        raise SystemExit(f"缺少模型权重：{weights_path}")

    model = resnet18(pretrained=False)
    model.eval()
    source_state = load_file(weights_path)
    target_state = model.state_dict()
    ignored = sorted(name for name in source_state if name.endswith("num_batches_tracked"))
    unexpected = sorted(
        name for name in source_state if name not in target_state and name not in ignored
    )
    missing = sorted(name for name in target_state if name not in source_state)
    shape_mismatches = sorted(
        name
        for name in target_state
        if name in source_state and tuple(target_state[name].shape) != tuple(source_state[name].shape)
    )
    if unexpected or missing or shape_mismatches:
        raise SystemExit(
            json.dumps(
                {
                    "unexpected": unexpected,
                    "missing": missing,
                    "shape_mismatches": shape_mismatches,
                },
                ensure_ascii=False,
            )
        )

    parameters = {name: source_state[name] for name in target_state}
    model.load_parameters(parameters)

    if args.input_evidence is not None:
        input_evidence = args.input_evidence.expanduser().resolve()
        if not input_evidence.is_file():
            raise SystemExit(f"输入证据不存在：{input_evidence}")
        with np.load(input_evidence, allow_pickle=False) as archive:
            if "input" not in archive:
                raise SystemExit(f"输入证据不包含 input：{input_evidence}")
            input_array = np.asarray(archive["input"], dtype=np.float32).copy()
        input_source = str(input_evidence)
    else:
        input_array = np.linspace(
            -1.0,
            1.0,
            num=3 * 224 * 224,
            dtype=np.float32,
        ).reshape(1, 3, 224, 224)
        input_source = "numpy.linspace"
    if input_array.shape != (1, 3, 224, 224):
        raise SystemExit(f"输入形状必须是 (1, 3, 224, 224)，实际为 {input_array.shape}")
    tensors: dict[str, np.ndarray] = {"input": input_array.copy()}
    summaries: dict[str, dict[str, Any]] = {}

    def capture(name: str, value: Any) -> None:
        array = np.asarray(value.numpy(), dtype=np.float32).copy()
        tensors[f"probe.{name}"] = array
        summaries[name] = array_summary(array)

    value = jt.array(input_array)
    value = model.conv1(value)
    capture("conv1", value)
    value = model.bn1(value)
    capture("bn1", value)
    value = model.relu(value)
    value = model.maxpool(value)
    capture("maxpool", value)

    def execute_stage(stage: Any, stage_name: str, stage_input: Any) -> Any:
        current = stage_input
        for block_index in range(2):
            block = stage[block_index]
            prefix = f"{stage_name}.{block_index}"
            identity = current
            output = block.conv1(current)
            capture(f"{prefix}.conv1", output)
            output = block.bn1(output)
            capture(f"{prefix}.bn1", output)
            output = block.relu(output)
            output = block.conv2(output)
            capture(f"{prefix}.conv2", output)
            output = block.bn2(output)
            capture(f"{prefix}.bn2", output)
            if block.downsample is not None:
                identity = block.downsample[0](current)
                capture(f"{prefix}.downsample.0", identity)
                identity = block.downsample[1](identity)
                capture(f"{prefix}.downsample.1", identity)
            output += identity
            current = block.relu(output)
            capture(prefix, current)
        capture(stage_name, current)
        return current

    value = execute_stage(model.layer1, "layer1", value)
    value = execute_stage(model.layer2, "layer2", value)
    value = execute_stage(model.layer3, "layer3", value)
    value = execute_stage(model.layer4, "layer4", value)
    value = model.avgpool(value).float_auto()
    value = jt.reshape(value, (value.shape[0], -1))
    value = model.fc(value)
    capture("fc", value)
    output_array = tensors["probe.fc"].copy()
    tensors["output"] = output_array

    args.tensors.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.tensors, **tensors)
    result = {
        "schema_version": 1,
        "framework": "jittor",
        "jittor_version": jt.__version__,
        "numpy_version": np.__version__,
        "device": args.device,
        "model_id": "cnn-resnet18",
        "model_class": type(model).__name__,
        "weights": {
            "path": str(weights_path),
            "loaded": len(parameters),
            "ignored": ignored,
        },
        "input": array_summary(input_array),
        "input_source": input_source,
        "output": array_summary(output_array),
        "probes": summaries,
        "tensor_evidence": str(args.tensors.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
