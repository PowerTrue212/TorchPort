from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

print("正在导入 timm 和 PyTorch……", flush=True)
import numpy as np  # noqa: E402 - 导入可能较慢，先向用户输出进度
import timm  # noqa: E402 - 导入可能较慢，先向用户输出进度
import torch  # noqa: E402 - 导入可能较慢，先向用户输出进度
print(f"已导入 timm {timm.__version__}、PyTorch {torch.__version__}", flush=True)


RESNET18_PROBES = (
    "conv1",
    "bn1",
    "maxpool",
    "layer1.0.conv1",
    "layer1.0.bn1",
    "layer1.0.conv2",
    "layer1.0.bn2",
    "layer1.0",
    "layer1.1.conv1",
    "layer1.1.bn1",
    "layer1.1.conv2",
    "layer1.1.bn2",
    "layer1.1",
    "layer1",
    "layer2.0.conv1",
    "layer2.0.bn1",
    "layer2.0.conv2",
    "layer2.0.bn2",
    "layer2.0.downsample.0",
    "layer2.0.downsample.1",
    "layer2.0",
    "layer2.1.conv1",
    "layer2.1.bn1",
    "layer2.1.conv2",
    "layer2.1.bn2",
    "layer2.1",
    "layer2",
    "layer3.0.conv1",
    "layer3.0.bn1",
    "layer3.0.conv2",
    "layer3.0.bn2",
    "layer3.0.downsample.0",
    "layer3.0.downsample.1",
    "layer3.0",
    "layer3.1.conv1",
    "layer3.1.bn1",
    "layer3.1.conv2",
    "layer3.1.bn2",
    "layer3.1",
    "layer3",
    "layer4.0.conv1",
    "layer4.0.bn1",
    "layer4.0.conv2",
    "layer4.0.bn2",
    "layer4.0.downsample.0",
    "layer4.0.downsample.1",
    "layer4.0",
    "layer4.1.conv1",
    "layer4.1.bn1",
    "layer4.1.conv2",
    "layer4.1.bn2",
    "layer4.1",
    "layer4",
    "fc",
)

VIT_TINY_PROBES = (
    "patch_embed",
    "embedding",
    *(
        name
        for block_index in range(12)
        for name in (
            f"blocks.{block_index}.norm1",
            f"blocks.{block_index}.attn.qkv",
            f"blocks.{block_index}.attn.scores",
            f"blocks.{block_index}.attn.probabilities",
            f"blocks.{block_index}.attn.context",
            f"blocks.{block_index}.attn.proj",
            f"blocks.{block_index}.attn",
            f"blocks.{block_index}.norm2",
            f"blocks.{block_index}.mlp.fc1",
            f"blocks.{block_index}.mlp.act",
            f"blocks.{block_index}.mlp.fc2",
            f"blocks.{block_index}.mlp",
            f"blocks.{block_index}",
        )
    ),
    "norm",
    "head",
)


MODEL_PROBES = {
    "cnn-resnet18": RESNET18_PROBES,
    "vit-tiny-patch16": VIT_TINY_PROBES,
}


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().contiguous().numpy().copy()


def array_summary(value: np.ndarray, *, dtype: str) -> dict[str, Any]:
    value = np.asarray(value, dtype=np.float32)
    statistics = value.astype(np.float64, copy=False)
    return {
        "shape": list(value.shape),
        "dtype": dtype,
        "mean": float(statistics.mean()),
        "std": float(statistics.std()),
        "min": float(statistics.min()),
        "max": float(statistics.max()),
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"输出中没有张量：{type(value)!r}")


def run_vit_tiny_manual(
    model: Any,
    value: torch.Tensor,
    record: Callable[[str, torch.Tensor], None],
) -> torch.Tensor:
    value = model.patch_embed(value)
    record("patch_embed", value)
    value = model._pos_embed(value)
    value = model.patch_drop(value)
    value = model.norm_pre(value)
    record("embedding", value)

    for block_index, block in enumerate(model.blocks):
        prefix = f"blocks.{block_index}"
        normalized = block.norm1(value)
        record(f"{prefix}.norm1", normalized)

        batch, tokens, _channels = normalized.shape
        attention = block.attn
        qkv = attention.qkv(normalized)
        record(f"{prefix}.attn.qkv", qkv)
        qkv_heads = qkv.reshape(
            batch,
            tokens,
            3,
            attention.num_heads,
            attention.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, item_value = qkv_heads.unbind(0)
        query, key = attention.q_norm(query), attention.k_norm(key)
        scores = (query * attention.scale) @ key.transpose(-2, -1)
        record(f"{prefix}.attn.scores", scores)
        probabilities = attention.attn_drop(scores.softmax(dim=-1))
        record(f"{prefix}.attn.probabilities", probabilities)
        attention_output = probabilities @ item_value
        attention_output = attention_output.transpose(1, 2).reshape(
            batch,
            tokens,
            attention.attn_dim,
        )
        attention_output = attention.norm(attention_output)
        record(f"{prefix}.attn.context", attention_output)
        attention_output = attention.proj(attention_output)
        record(f"{prefix}.attn.proj", attention_output)
        attention_output = attention.proj_drop(attention_output)
        record(f"{prefix}.attn", attention_output)
        value = value + block.drop_path1(block.ls1(attention_output))

        normalized = block.norm2(value)
        record(f"{prefix}.norm2", normalized)
        mlp_output = block.mlp.fc1(normalized)
        record(f"{prefix}.mlp.fc1", mlp_output)
        mlp_output = block.mlp.act(mlp_output)
        record(f"{prefix}.mlp.act", mlp_output)
        mlp_output = block.mlp.drop1(mlp_output)
        mlp_output = block.mlp.norm(mlp_output)
        mlp_output = block.mlp.fc2(mlp_output)
        record(f"{prefix}.mlp.fc2", mlp_output)
        mlp_output = block.mlp.drop2(mlp_output)
        record(f"{prefix}.mlp", mlp_output)
        value = value + block.drop_path2(block.ls2(mlp_output))
        record(prefix, value)

    value = model.norm(value)
    record("norm", value)
    output = model.head(model.head_drop(model.fc_norm(value[:, 0])))
    record("head", output)
    return output


def run_model(
    model_id: str,
    model_root: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    print(f"正在加载模型：{model_id}", flush=True)
    model_directory = (model_root / model_id).resolve()
    if not (model_directory / "model.safetensors").is_file():
        raise FileNotFoundError(f"缺少模型权重：{model_directory}")

    model = timm.create_model(f"local-dir:{model_directory}", pretrained=True)
    model.eval().to(device)
    runtime_options: dict[str, Any] = {}
    if model_id == "vit-tiny-patch16":
        for block in model.blocks:
            block.attn.fused_attn = False
        runtime_options["fused_attention"] = False

    captured: dict[str, dict[str, Any]] = {}
    tensors: dict[str, np.ndarray] = {}
    handles = []

    def record(probe_name: str, output: Any) -> None:
        tensor = first_tensor(output)
        array = tensor_to_numpy(tensor)
        tensors[f"probe.{probe_name}"] = array
        captured[probe_name] = array_summary(array, dtype=str(tensor.dtype))

    if model_id != "vit-tiny-patch16":
        modules = dict(model.named_modules())
        for name in MODEL_PROBES[model_id]:
            module = modules.get(name)
            if module is None:
                continue

            def capture(_module, _inputs, output, *, probe_name=name):
                record(probe_name, output)

            handles.append(module.register_forward_hook(capture))

    input_tensor = torch.linspace(
        -1.0,
        1.0,
        steps=3 * 224 * 224,
        dtype=torch.float32,
        device=device,
    ).reshape(1, 3, 224, 224)

    try:
        print(f"正在执行前向：{model_id}，设备={device}", flush=True)
        with torch.inference_mode():
            if model_id == "vit-tiny-patch16":
                output = run_vit_tiny_manual(model, input_tensor, record)
            else:
                output = first_tensor(model(input_tensor))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()

    input_array = tensor_to_numpy(input_tensor)
    output_array = tensor_to_numpy(output)
    ordered_tensors = {"input": input_array, **tensors, "output": output_array}
    print(f"参考结果生成完成：{model_id}", flush=True)
    return {
        "model_id": model_id,
        "model_class": type(model).__name__,
        "runtime_options": runtime_options,
        "input": array_summary(input_array, dtype=str(input_tensor.dtype)),
        "output": array_summary(output_array, dtype=str(output.dtype)),
        "probes": captured,
    }, ordered_tensors


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 PyTorch 模型参考证据")
    parser.add_argument("--models-root", type=Path, default=Path("artifacts/models"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/pytorch-reference/reference.json"),
    )
    parser.add_argument(
        "--tensors-directory",
        type=Path,
        default=Path("artifacts/runs/pytorch-reference/tensors"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_PROBES),
        default=list(MODEL_PROBES),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("请求使用 CUDA，但 PyTorch 没有检测到可用设备。")

    torch.manual_seed(20260818)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260818)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    model_results = []
    args.tensors_directory.mkdir(parents=True, exist_ok=True)
    for model_id in args.models:
        model_result, tensors = run_model(model_id, args.models_root, device)
        tensor_path = (args.tensors_directory / f"{model_id}.npz").resolve()
        np.savez(tensor_path, **tensors)
        model_result["tensor_evidence"] = str(tensor_path)
        model_results.append(model_result)

    result = {
        "schema_version": 1,
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "seed": 20260818,
        "models": model_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
