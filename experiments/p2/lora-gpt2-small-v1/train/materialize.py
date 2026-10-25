"""Materialize the GPT-2 Small + LoRA train-step calibration task.

Builds two task repositories outside Git (`migrations/LoRA/train/`):

- `source/`: the frozen Microsoft LoRA tree plus a thin `torchport_train.py` entry that calls the
  repository's own model, AdamW and linear warmup schedule.
- `reference-target/`: the paired manual Jittor port's model code plus a faithful Jittor entry
  (source-equivalent AdamW and LambdaLR semantics). It is the known-correct calibration candidate,
  never shown to an Agent.

Also freezes the token batches and converts the GPT-2 base checkpoint plus the fixture LoRA state
into one NPZ whose keys match the source model's state_dict. Needs a local PyTorch (CPU is enough);
nothing is executed on GPU here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MIGRATION = ROOT / "migrations" / "LoRA"
SOURCE = MIGRATION / "source" / "microsoft-LoRA"
REFERENCE = MIGRATION / "reference" / "LoRA-GPT2-E2E-pytorch-jittor" / "by-jittor" / "NLG" / "src"
OUT = MIGRATION / "train"
BASE_CHECKPOINT = MIGRATION / "checkpoint" / "gpt2-pytorch_model.bin"
BASE_CHECKPOINT_SHA256 = "7c5d3f4b8b76583b422fcb9189ad6c89d5d97a094541ce8932dce3ecabde1421"
LORA_STATE = HERE.parent / "fixtures" / "lora_state.npz"
WEIGHTS = MIGRATION / "checkpoint" / "gpt2-sm-lora-train.npz"
FIXTURES = HERE / "fixtures"
SEED = 20261020
STEPS, BATCH, SEQ = 4, 2, 16

# 训练步只用 NLG/src 与 loralib；数据集、NLU、评测脚本和图片不进入 task 仓库，避免同步上百 MB。
IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", "*.pyc", "pretrained_checkpoints", "trained_models",
    "data", "NLU", "eval", "figures",
)

# 两端共用的冻结超参；与 E2E 官方命令一致（correct_bias、beta2=0.98、eps=1e-6、不裁剪），
# 只缩短 warmup/max_step 让 4 步内覆盖 lr=0、warmup 和线性衰减。
HYPER = '''LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_STEP = 2
MAX_STEP = 6
'''

SOURCE_ENTRY = '''"""TorchPort train-step entry for the frozen Microsoft LoRA source. Calls repository code only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "examples" / "NLG" / "src"))

import loralib as lora  # noqa: E402
import torch  # noqa: E402
from model import GPT2Config, GPT2LMModel  # noqa: E402
from optimizer import create_adam_optimizer, get_linear_schedule_with_warmup  # noqa: E402

''' + HYPER + '''

def build_model():
    config = GPT2Config(
        n_embd=768, n_layer=12, n_head=12, lora_attn_dim=4, lora_attn_alpha=32, lora_dropout=0.0
    )
    model = GPT2LMModel(config)
    lora.mark_only_lora_as_trainable(model)
    return model


class TorchPortLoRAGPT2Small(torch.nn.Module):
    """推理任务入口：与 candidate 的 torchport_adapter 同名同结构（参数名带 model. 前缀），只返回 logits。"""

    def __init__(self):
        super().__init__()
        self.model = build_model()

    def forward(self, input_ids):
        return (self.model(input_ids)[0],)


def build_optimizer(model):
    optimizer = create_adam_optimizer(model, LR, WEIGHT_DECAY, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEP, MAX_STEP)
    return optimizer, scheduler


def train_step(model, state, input_ids, labels, mask):
    # 与 gpt2_ft.optimizer_step（fp16=False、clip=0、grad_acc=1）相同的顺序
    optimizer, scheduler = state
    _, loss = model(input_ids, lm_labels=labels, lm_mask=mask)
    loss = loss.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    return loss.detach()
'''

TARGET_ENTRY = '''"""Faithful Jittor train-step entry for calibration (reference model code, source optimizer semantics)."""
from __future__ import annotations

import math

import jittor as jt

import loralib as lora
import model as _model
from model import GPT2Config, GPT2LMModel

''' + HYPER + '''
# 人工迁移的 gelu 用 jt.nn.gelu（erf 精确式），source 用 tanh 近似；V100 首轮 train-step
# 检出 loss 差约 1e-3（ratio 2146）。校准 candidate 恢复 source 的 tanh 公式；MLP 构造时
# 读取模块级 gelu，因此必须在 build_model 之前替换。
def _gelu_tanh(x):
    return 0.5 * x * (1 + jt.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * jt.pow(x, 3))))


_model.gelu = _gelu_tanh

# 人工迁移把 enable_lora / lora_ind 存成 jt.Var 属性，Jittor 会把它们算进 state_dict，
# 而 source 中它们是普通张量属性、不是参数。这里改存到下划线属性（Jittor 不收集），
# 用类属性 property 保持原代码读取方式；数值路径不变。
for _name in ("enable_lora", "lora_ind"):
    setattr(lora.MergedLinear, _name, property(
        lambda self, _n=_name: self.__dict__["_" + _n],
        lambda self, value, _n=_name: self.__dict__.__setitem__("_" + _n, value),
    ))


def build_model():
    config = GPT2Config(
        n_embd=768, n_layer=12, n_head=12, lora_attn_dim=4, lora_attn_alpha=32, lora_dropout=0.0
    )
    model = GPT2LMModel(config)
    lora.mark_only_lora_as_trainable(model)
    return model


class AdamW(jt.optim.Optimizer):
    """Microsoft LoRA 的 AdamW：eps 加在未校正的 sqrt(v) 上，衰减在更新之后。"""

    def __init__(self, params, lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0, correct_bias=True):
        super().__init__(params, lr)
        self.betas, self.eps = betas, eps
        self.weight_decay, self.correct_bias = weight_decay, correct_bias
        for group in self.param_groups:
            group["exp_avg"] = [jt.zeros(p.shape, p.dtype).stop_grad() for p in group["params"]]
            group["exp_avg_sq"] = [jt.zeros(p.shape, p.dtype).stop_grad() for p in group["params"]]

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        beta1, beta2 = self.betas
        n = self.n_step
        for group in self.param_groups:
            lr = group.get("lr", self.lr)
            step_size = lr
            if self.correct_bias:
                step_size = lr * math.sqrt(1.0 - beta2**n) / (1.0 - beta1**n)
            for p, g, m, v in zip(group["params"], group["grads"], group["exp_avg"], group["exp_avg_sq"]):
                if p.is_stop_grad():
                    continue
                m.update(beta1 * m + (1.0 - beta1) * g)
                v.update(beta2 * v + (1.0 - beta2) * g * g)
                p.update(p - step_size * m / (jt.sqrt(v) + self.eps))
                if self.weight_decay > 0.0:
                    p.update(p - lr * self.weight_decay * p)
        if hasattr(self, "post_step"):
            self.post_step()


class LinearWarmup:
    """torch LambdaLR 语义：构造时即设 lr=base*lambda(0)，每次 step 后 last_epoch+1。"""

    def __init__(self, optimizer, warmup, total):
        self.optimizer, self.warmup, self.total = optimizer, warmup, total
        self.base = [group.get("lr", optimizer.lr) for group in optimizer.param_groups]
        self.last_epoch = 0
        self._apply()

    def _factor(self, step):
        if step < self.warmup:
            return float(step) / float(max(1, self.warmup))
        return max(0.0, float(self.total - step) / float(max(1, self.total - self.warmup)))

    def _apply(self):
        for group, base in zip(self.optimizer.param_groups, self.base):
            group["lr"] = base * self._factor(self.last_epoch)

    def step(self):
        self.last_epoch += 1
        self._apply()


class TorchPortLoRAGPT2Small(jt.Module):
    """推理任务入口：与 source 同名同结构（参数名带 model. 前缀），只返回 logits。"""

    def __init__(self):
        super().__init__()
        self.model = build_model()

    def execute(self, input_ids):
        return (self.model(input_ids)[0],)


def build_optimizer(model):
    params = [p for _, p in model.named_parameters() if not p.is_stop_grad()]
    optimizer = AdamW(params, LR, weight_decay=WEIGHT_DECAY, correct_bias=True)
    return optimizer, LinearWarmup(optimizer, WARMUP_STEP, MAX_STEP)


def train_step(model, state, input_ids, labels, mask):
    optimizer, scheduler = state
    _, loss = model(input_ids, lm_labels=labels, lm_mask=mask)
    loss = loss.mean()
    optimizer.step(loss)
    scheduler.step()
    return loss
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_repo(source: Path, dest: Path, entry: str, force: bool) -> None:
    if dest.exists():
        if not force:
            raise FileExistsError(f"{dest} exists; use --force")
        shutil.rmtree(dest)
    shutil.copytree(source, dest, ignore=IGNORE)
    (dest / "torchport_train.py").write_text(entry, encoding="utf-8")


def write_inputs() -> list[dict]:
    rng = np.random.default_rng(SEED)
    tokens = rng.integers(0, 50_000, size=(STEPS, BATCH, SEQ + 1), dtype=np.int64)
    mask = np.ones((STEPS, BATCH, SEQ), dtype=np.float32)
    mask[:, 1, -3:] = 0.0  # 第二条序列尾部 padding，覆盖 lm_mask 归一化
    FIXTURES.mkdir(parents=True, exist_ok=True)
    specs = []
    for name, value in (("input_ids", tokens[..., :-1]), ("labels", tokens[..., 1:]), ("mask", mask)):
        path = FIXTURES / f"{name}.npy"
        np.save(path, np.ascontiguousarray(value), allow_pickle=False)
        specs.append({"shape": list(value.shape), "dtype": str(value.dtype), "path": path})
    return specs


def write_weights(source_repo: Path) -> None:
    import torch

    if sha256_file(BASE_CHECKPOINT) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("GPT-2 base checkpoint SHA-256 mismatch")
    sys.path[:0] = [str(source_repo), str(source_repo / "examples" / "NLG" / "src")]
    import torchport_train  # noqa: PLC0415

    model = torchport_train.build_model()
    model.load_weight(torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=True))
    state = model.state_dict()
    with np.load(LORA_STATE, allow_pickle=False) as lora_state:
        for name in lora_state.files:
            if name not in state or tuple(state[name].shape) != lora_state[name].shape:
                raise RuntimeError(f"LoRA fixture key does not match source model: {name}")
            state[name].copy_(torch.from_numpy(np.asarray(lora_state[name])))
    np.savez(WEIGHTS, **{name: value.detach().cpu().numpy() for name, value in state.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source_repo, target_repo = OUT / "source", OUT / "reference-target"
    write_repo(SOURCE, source_repo, SOURCE_ENTRY, args.force)
    write_repo(REFERENCE, target_repo, TARGET_ENTRY, args.force)
    inputs = write_inputs()
    write_weights(source_repo)

    # 远端打包按项目根解析相对路径，而 load_task 按 task 文件目录解析；用绝对路径避免两者不一致。
    side = {"entry": "torchport_train.py", "model": "build_model", "device": "cuda"}
    task = {
        "schema_version": 1,
        "task_id": "lora-gpt2-small-train-step-v1",
        "source": {"repo": str(source_repo), **side},
        "target": {"repo": str(target_repo), **side},
        "inputs": [{**item, "path": str(item["path"])} for item in inputs],
        "weights": str(WEIGHTS),
        "seed": SEED,
        "protocol": {
            "profile": "train-step",
            "repeats": 2,
            "timeout": 1800,
            "train": {"optimizer": "build_optimizer", "step": "train_step"},
        },
    }
    (HERE / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", "utf-8")

    # 推理任务：同一权重、同一入口文件，模型换成只返回 logits 的包装类；固定 token batch 取
    # 已冻结的 fixtures/input_ids.npy（2x16），与训练 batch 无关。
    infer_side = {**side, "model": "TorchPortLoRAGPT2Small"}
    infer_input = HERE.parent / "fixtures" / "input_ids.npy"
    infer = {
        "schema_version": 1,
        "task_id": "lora-gpt2-small-forward-v1",
        "source": {"repo": str(source_repo), **infer_side},
        "target": {"repo": str(target_repo), **infer_side},
        "inputs": [{"shape": [2, 16], "dtype": "int64", "path": str(infer_input)}],
        "weights": str(WEIGHTS),
        "seed": SEED,
        "protocol": {"profile": "fp32-inference", "repeats": 2, "timeout": 1800},
    }
    (HERE / "task-forward.json").write_text(
        json.dumps(infer, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print(json.dumps({"task": str(HERE / "task.json"), "forward_task": str(HERE / "task-forward.json"),
                      "weights_sha256": sha256_file(WEIGHTS)}, indent=2))


if __name__ == "__main__":
    main()
