from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行原生 Jittor ViT-Tiny 并采集分层证据")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("artifacts/models/vit-tiny-patch16/model.safetensors"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/jittor-vit-tiny/evidence.json"),
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        default=Path("artifacts/runs/jittor-vit-tiny/vit-tiny-patch16.npz"),
    )
    parser.add_argument(
        "--input-evidence",
        type=Path,
        help="可选的源框架 NPZ 证据；提供时读取其中的 input",
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
        os.environ["nvcc_path"] = ""

    import numpy as np
    import jittor as jt
    from safetensors.numpy import load_file

    from torchport.adapters.jittor import VisionTransformerTiny

    if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
        raise SystemExit(
            "Jittor 1.3.11.0 需要项目隔离的 NumPy 1.x；"
            "请把 artifacts/jittor-deps 放在 PYTHONPATH 首位。"
        )
    if args.device == "cuda":
        jt.flags.use_cuda = 1

    weights_path = args.weights.expanduser().resolve()
    if not weights_path.is_file():
        raise SystemExit(f"缺少模型权重：{weights_path}")

    model = VisionTransformerTiny()
    model.eval()
    source_state = load_file(weights_path)
    target_state = model.state_dict()
    unexpected = sorted(name for name in source_state if name not in target_state)
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
    print(f"已加载 {len(parameters)} 个 ViT-Tiny 权重", flush=True)

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
        array = np.asarray(value.fetch_sync(), dtype=np.float32).copy()
        tensors[f"probe.{name}"] = array
        summaries[name] = array_summary(array)

    value = jt.array(input_array)
    value = model.patch_embed(value)
    capture("patch_embed", value)
    class_token = model.cls_token.repeat(value.shape[0], 1, 1)
    value = jt.concat((class_token, value), dim=1) + model.pos_embed
    capture("embedding", value)

    for block_index in range(12):
        print(f"正在执行 Transformer 块 {block_index + 1}/12", flush=True)
        block = model.blocks[block_index]
        prefix = f"blocks.{block_index}"

        normalized = block.norm1(value)
        capture(f"{prefix}.norm1", normalized)

        batch, tokens, channels = normalized.shape
        qkv = block.attn.qkv(normalized)
        capture(f"{prefix}.attn.qkv", qkv)
        qkv_heads = qkv.reshape(
            (batch, tokens, 3, block.attn.num_heads, block.attn.head_dim)
        ).transpose(2, 0, 3, 1, 4)
        query, key, item_value = qkv_heads[0], qkv_heads[1], qkv_heads[2]
        scores = jt.matmul(
            query * block.attn.scale,
            key.transpose(0, 1, 3, 2),
        )
        capture(f"{prefix}.attn.scores", scores)
        probabilities = scores.softmax(dim=-1)
        capture(f"{prefix}.attn.probabilities", probabilities)
        attention_output = jt.matmul(probabilities, item_value)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            (batch, tokens, channels)
        )
        capture(f"{prefix}.attn.context", attention_output)
        attention_output = block.attn.proj(attention_output)
        capture(f"{prefix}.attn.proj", attention_output)
        capture(f"{prefix}.attn", attention_output)
        value = value + attention_output

        normalized = block.norm2(value)
        capture(f"{prefix}.norm2", normalized)
        mlp_output = block.mlp.fc1(normalized)
        capture(f"{prefix}.mlp.fc1", mlp_output)
        mlp_output = block.mlp.act(mlp_output)
        capture(f"{prefix}.mlp.act", mlp_output)
        mlp_output = block.mlp.fc2(mlp_output)
        capture(f"{prefix}.mlp.fc2", mlp_output)
        capture(f"{prefix}.mlp", mlp_output)
        value = value + mlp_output
        capture(prefix, value)

    value = model.norm(value)
    capture("norm", value)
    output = model.head(value[:, 0])
    capture("head", output)
    output_array = tensors["probe.head"].copy()
    tensors["output"] = output_array

    args.tensors.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.tensors, **tensors)
    result = {
        "schema_version": 1,
        "framework": "jittor",
        "jittor_version": jt.__version__,
        "numpy_version": np.__version__,
        "device": args.device,
        "model_id": "vit-tiny-patch16",
        "model_class": type(model).__name__,
        "weights": {
            "path": str(weights_path),
            "loaded": len(parameters),
        },
        "input": array_summary(input_array),
        "input_source": input_source,
        "output": array_summary(output_array),
        "probes": summaries,
        "tensor_evidence": str(args.tensors.resolve()),
        "runtime_options": {
            "fused_attention": False,
            "layer_norm": "centered-variance",
        },
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
