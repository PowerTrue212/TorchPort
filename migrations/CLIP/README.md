# CLIP PyTorch → Jittor 三方法对比（冻结版）

这个目录只实验一件事：把同一个 OpenAI CLIP ViT-B/32 分别交给
Codex、DSH、DSH + TorchPort 迁移到 Jittor，然后用同一个 evaluator 比较最终
`4 × 4` image-text logits 与 PyTorch 参考值是否对齐。

不训练，不测 FP16，不迁移 tokenizer，不测梯度，不增加模型、输入、指标或图。

## 已冻结参数

| 项目 | 固定值 |
|---|---|
| 上游仓库 | `openai/CLIP` |
| commit | `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |
| 模型 | `ViT-B/32` |
| 权重 SHA256 | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |
| 输入 | CIFAR-10 test 固定索引 `12/0/6/3`，顺序 dog/cat/car/airplane |
| 文本 | `a photo of a dog/cat/car/an airplane` |
| 硬件 | 同一台 NVIDIA V100，CUDA |
| 精度 | FP32（参考模型显式 `model.float()`） |
| 源环境 | Python 3.10，PyTorch `2.6.0+cu124`，torchvision `0.21.0+cu124` |
| 目标环境 | Python 3.9，Jittor `1.3.11.0`，NumPy `1.26.4` |
| Agent 环境 | Node `22.23.2`，DSH `0.1.0-rc.7`，plugin API `0.1.0-rc.8` |
| NA 判定 | `np.allclose(rtol=1e-4, atol=1e-4, equal_nan=False)` |
| 最大修复轮数 | 8 |

全部机器可读参数只有一份：[`experiment.json`](experiment.json)。

`Rounds` 的定义是：初次迁移尝试为 0；每次 common evaluator 失败后，Agent
根据证据做一组连贯修复，记 1 轮。每轮必须在修改前通过
`record_round.py` 写入 JSONL；第 8 轮后无论成败都停止。

## 目录与生成产物

```text
migrations/CLIP/
├── experiment.json              # 唯一冻结参数
├── prompts/                     # 三方法独立提示词和权限边界
├── scripts/                     # 准备、建工作区、计轮、评估、画图
├── templates/submission.py      # 三方法相同的 Jittor 接口
├── source/                      # 生成：固定 commit 的官方 CLIP
├── shared/                      # 生成：权重、固定输入、参考 logits、哈希
├── workspaces/{method}/repo/    # 生成：三份完全相同的起点
├── results/{method}/            # 生成：每轮评估和最终 logits
└── outputs/                     # 生成：结果表和且仅有两张图
```

`weights_fp32.npz` 是一次性由官方 `.pt` checkpoint 导出的中性 FP32 state
archive，三种方法共用。它不包含迁移代码，也不参与指标计算；这样
Jittor 环境无需安装 PyTorch 即可读取同一份权重。

## V100 上的准备

从 TorchPort 根目录上传整个 `migrations/CLIP` 目录后：

```bash
cd /data/L202500448/torchport/CLIP
cp env.example.sh env.sh
# 确认 env.sh 中 SOURCE_PYTHON / JITTOR_PYTHON 实际存在

"$SOURCE_PYTHON" -m pip install -r requirements/source.txt
"$SOURCE_PYTHON" -m pip install -r requirements/viz.txt
"$JITTOR_PYTHON" -m pip install -r requirements/target.txt

"$SOURCE_PYTHON" -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__)'
"$JITTOR_PYTHON" -c 'import jittor, numpy; print(jittor.__version__, numpy.__version__)'

bash run_prepare.sh
```

`run_prepare.sh` 按顺序完成：

1. clone 官方 CLIP 并 detached checkout 固定 commit；
2. 下载并核验官方 ViT-B/32 权重和 CIFAR-10 archive；
3. 只提取四张固定 PNG；
4. 用官方 preprocessing/tokenizer 生成 `images.npy` 和 `tokens.npy`；
5. 用 V100 FP32 生成 `logits_ref.npy` 和 `weights_fp32.npz`；
6. 核验形状、dtype、commit 和所有 SHA256；
7. 复制三份起始内容完全相同的 workspace。

脚本对已存在的 workspace 拒绝覆盖，避免误删 Agent 迁移轨迹。

## 运行三次迁移

三份 prompt 分别是：

- [`prompts/codex.md`](prompts/codex.md)
- [`prompts/dsh.md`](prompts/dsh.md)
- [`prompts/dsh_torchport.md`](prompts/dsh_torchport.md)

把 Agent 工作目录分别指向对应 `workspaces/<method>/repo`。Codex 和 DSH prompt
明确禁止 TorchPort；只有 DSH + TorchPort 能读取插件证据。三者都只能修改
自己的 workspace，都必须实现相同的 `submission.py` 接口。

每个方法最终都由相同脚本评估：

```bash
bash run_evaluate.sh codex
bash run_evaluate.sh dsh
bash run_evaluate.sh dsh_torchport
```

Evaluator 在隔离的 Jittor 环境中执行，并强制：

- CUDA 可用；
- `load_model` 返回 `jittor.Module`；
- `forward_logits` 返回 `jittor.Var[4,4]`；
- 输出全部 finite；
- submission 不能在 evaluator 中导入 PyTorch；
- 准备文件的 SHA256 没有被改动。

RN = forward 成功且输出合法；NA = 通过冻结的 `np.allclose`；Max Error =
`max(abs(logits_ref - logits_jittor))`。TorchPort 的 SA/FD 只留在该方法轨迹中。

## 生成最终结果

三个方法都停止后：

```bash
bash run_report.sh
```

结果只有两张图：

```text
outputs/heatmaps.png       # PyTorch | Codex | DSH | DSH + TorchPort，共用色域
outputs/method_table.png   # Method | RN | NA | Max Error | Rounds
```

同时保留便于论文制表和审计的
`outputs/method_results.csv`、`outputs/method_results.md` 和 `outputs/summary.json`，
但不产生其他可视化。

## 上传前的本地静态检查

这一步不需要 GPU：

```bash
python -m compileall -q scripts templates tests
python -m pytest migrations/CLIP/tests -q
```

真正的 reference 和 Jittor forward 必须到同一台 V100 上运行，本地 CPU 结果不计入
实验。
