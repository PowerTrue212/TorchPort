# APT PyTorch → Jittor 三方法完整训练对比（冻结版）

这个目录按 TorchPort 现有 `CLIP/` 的实验骨架组织，但实验语义不同：CLIP 是固定权重
的一次推理对齐；APT 必须从 ImageNet-21k 预训练 ViT-B/16 开始，依次完成 CIFAR-100
的 10 个 class-incremental tasks，每个 task 训练 30 epochs，最后再对完整 10,000 张
测试集推理。

三种迁移方法仍然是：

```text
同一份冻结 APT source
├── Codex          → Jittor A（禁止 TorchPort）
├── DSH            → Jittor B（禁止 TorchPort）
└── DSH+TorchPort  → Jittor C（允许 TorchPort 诊断）
```

## 为什么流程不能照抄 CLIP

APT 的结果取决于完整训练轨迹，因此不使用 `logits allclose / Max Error`。common
evaluator 要求每个方法提交完整的 `R[t,i]` accuracy matrix，并按原论文定义统一计算：

- **Average Accuracy**：最终模型在 10 个任务上的平均准确率；
- **Forgetting**：前 9 个任务从刚学完到最终模型的平均准确率下降；
- **Delta Accuracy**：Jittor Average Accuracy 减去 PyTorch reference；
- **RN**：10-task 训练与最终 10,000-image inference 是否完整运行；
- **Rounds**：Agent 在 common evaluator 失败后修复了几轮。

Inference 不提供 task identity。三种方法的指标只由 common evaluator 根据
`accuracy_matrix.npy` 和 `predictions.npz` 重新计算。

## 冻结参数

| 项目 | 固定值 |
|---|---|
| Source | 官方 `HaoranChen/Additive-Prompt-Tuning` commit `9b293da`；本地 ZIP SHA256 `5a27409a...f91f6fb` |
| Dataset | CIFAR-100，10 tasks × 10 classes，无 replay memory |
| Backbone | timm ViT-B/16，ImageNet-21k pretrained |
| Training | 30 epochs/task，batch 64，Adam，lr 0.004，FP32 |
| APT | prompt dropout 0.01，PPF/EMA coefficient 0.7 |
| Seed | 1 |
| Hardware | 同一台 NVIDIA V100，GPU 0 |
| Source env | Python 3.10，PyTorch 2.4.1，torchvision 0.19.1 |
| Target env | Python 3.9，Jittor 1.3.11.0 |
| 最大修复轮数 | 8 |

全部机器可读参数只在 [`experiment.json`](experiment.json) 中维护。

## 与 CLIP 同构的目录

```text
apt/
├── assets/                       # 固定 APT source ZIP + CIFAR-100 archive
├── experiment.json               # 唯一冻结参数
├── prompts/                      # Codex / DSH / DSH+TorchPort 权限边界
├── requirements/                 # PyTorch source / Jittor target / report 环境
├── scripts/                      # bootstrap/reference/workspace/evaluator/report
├── templates/submission.py       # 三方法相同的提交接口
├── source/apt/                   # 生成：协议适配后的相同起点
├── shared/data/                  # 生成：固定 CIFAR-100
├── shared/checkpoints/           # 生成：固定 ViT-B/16 checkpoint
├── shared/reference/             # 生成：PyTorch accuracy matrix + final predictions
├── workspaces/{method}/repo/     # 生成：三份完全相同的 source
├── results/{method}/             # 生成：训练、推理、rounds 和 common metrics
└── outputs/                      # 生成：结果表和两张核心图
```

`bootstrap.py` 对所有方法共同做且只做三项协议适配：预训练权重改为显式本地路径、
CLI seed 真正控制随机数、测试 DataLoader 使用 `drop_last=False`。这些适配在复制三份
workspace 之前完成，因此不属于任何迁移方法的优势。

## V100 上的流程

计划上传路径是 `/data-fs/APT`。V100 重新打开后先确认挂载点真实存在，再上传。

```bash
cd /data-fs/APT
cp env.example.sh env.sh
# 设置 SOURCE_PYTHON 和 JITTOR_PYTHON
```

两个解释器必须隔离。若服务器还没有环境，可按下面创建：

```bash
python3.10 -m venv /data-fs/APT/.venv-source
/data-fs/APT/.venv-source/bin/python -m pip install --upgrade pip
/data-fs/APT/.venv-source/bin/python -m pip install \
  torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
/data-fs/APT/.venv-source/bin/python -m pip install -r requirements/source.txt

python3.9 -m venv /data-fs/APT/.venv-jittor
/data-fs/APT/.venv-jittor/bin/python -m pip install --upgrade pip
/data-fs/APT/.venv-jittor/bin/python -m pip install -r requirements/target.txt
```

然后在 `env.sh` 中设置：

```bash
export SOURCE_PYTHON=/data-fs/APT/.venv-source/bin/python
export JITTOR_PYTHON=/data-fs/APT/.venv-jittor/bin/python
```

### 1. Bootstrap + 三份相同 workspace

```bash
bash run_prepare.sh
```

它会校验 source/data 哈希、下载并校验 ViT checkpoint、生成 `source/shared`，然后创建
三份完全相同的 workspace。已存在 workspace 时拒绝覆盖。

### 2. 完整 PyTorch reference training

```bash
bash run_reference.sh
```

这是 APT 相比 CLIP 新增的独立阶段：完整训练 10 tasks，保留 task checkpoints，生成
PyTorch `10×10` accuracy matrix 和最终 10,000 张测试图预测。没有 reference 时 common
Jittor evaluator 不会运行。

### 3. 三种迁移

分别把 Agent 工作目录指到：

- `workspaces/codex/repo`
- `workspaces/dsh/repo`
- `workspaces/dsh_torchport/repo`

对应提示词位于 `prompts/`。每个方法必须实现相同的
`submission.py::run_experiment`，完成自己的 10-task Jittor 训练和最终推理。

### 4. Common evaluator

```bash
bash run_evaluate.sh codex
bash run_evaluate.sh dsh
bash run_evaluate.sh dsh_torchport
```

Evaluator 验证 Jittor/V100、禁止 final submission 使用 PyTorch，并强制检查：

- `accuracy_matrix.npy`: float32 `[10,10]`，下三角有效，上三角 NaN；
- `predictions.npz`: float32 logits `[10000,100]`；
- targets/predictions/task_ids: int64 `[10000]`；
- 每个 task 恰好 1,000 张测试图；
- predictions 等于 logits argmax；
- 最终预测准确率与 accuracy matrix 最后一行一致。

### 5. 报告

```bash
bash run_report.sh
```

只生成两张核心图：

```text
outputs/accuracy_matrices.png  # PyTorch | Codex | DSH | DSH+TorchPort，统一全方法色域
outputs/method_table.png       # RN | Avg Acc | Forgetting | Delta Acc | Rounds
```

`accuracy_matrices.png` 的横轴是被评估任务 `i`，纵轴是刚训练完的任务 `t`。对角线表示
任务刚学完时的性能，最后一行表示最终模型，二者的下降直接对应 Forgetting。

## 本地静态检查

本地不运行训练，只检查协议和代码：

```powershell
python -m compileall -q scripts templates tests
python -m pytest apt/tests -q
python apt/scripts/validate_protocol.py --stage assets
```
