COPY_SKIP = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "build",
    "dist",
    "artifacts",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

TORCH_DEPS = {"torch", "torchvision", "torchaudio", "torchtext"}

BLOCKED_MODS = (
    "torch.autograd",
    "torch.distributed",
    "torch.fx",
    "torch.jit",
    "torch.library",
    "torch.onnx",
    "torch.profiler",
    "torch.utils.cpp_extension",
)

RISK_DEPS = {
    "flash_attn",
    "torch_geometric",
    "torchaudio",
    "torchtext",
    "torchvision",
    "torchvision.ops",
    "triton",
    "xformers",
}

TORCH_ATTRS = {
    "cat": "concat",
    "concat": "concat",
    "from_numpy": "array",
    "as_tensor": "array",
    "tensor": "array",
    "inference_mode": "no_grad",
    "manual_seed": "set_global_seed",
    "Tensor": "Var",
}

NN_ATTRS = {
    "Conv1d": "Conv1d",
    "Conv2d": "Conv",
    "Conv3d": "Conv3d",
    "ConvTranspose2d": "ConvTranspose",
    "BatchNorm1d": "BatchNorm1d",
    "BatchNorm2d": "BatchNorm",
    "BatchNorm3d": "BatchNorm3d",
}

TORCH_NAMES = {
    "Tensor": "Var",
    "as_tensor": "array",
    "cat": "concat",
    "concat": "concat",
    "from_numpy": "array",
    "inference_mode": "no_grad",
    "manual_seed": "set_global_seed",
    "tensor": "array",
}

JT_OPTIMS = {"Adan", "Adam", "AdamW", "LambdaLR", "RMSprop", "SGD"}

PORT_OPTIMS = {"Adadelta"}
