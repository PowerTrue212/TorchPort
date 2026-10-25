# CLIP ViT-B/32 迁移 pilot

本目录准备 OpenAI CLIP 官方仓库的 Jittor 迁移起点。任务使用真实预训练权重和 CIFAR-10
测试数据。none 和 full 使用相同的 Git 提交与提示词。两组只在实验控制器提供的反馈层级上不同。

## 固定材料

| 材料 | 位置或版本 |
| --- | --- |
| 官方源码 | `openai/CLIP@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |
| Jittor 初始仓库 | `D:\科研\CLIP-Benchmark-v1\seed\openai-CLIP`，提交 `6ee81bab4477fc10731b8e76f626d35f239b005c` |
| none 工作区 | `D:\科研\CLIP-Benchmark-v1\workspaces\none` |
| full 工作区 | `D:\科研\CLIP-Benchmark-v1\workspaces\full` |
| 官方 ViT-B/32 checkpoint | `migrations/CLIP/checkpoint/ViT-B-32.pt`，SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |
| CIFAR-10 Python archive | `migrations/CLIP/data/cifar-10-python.tar.gz`，SHA-256 `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce` |
| Harness FP32 权重 | `migrations/CLIP/checkpoint/weights_fp32.npz`，302 项，SHA-256 `d89c2d139d6398b23dceef40ec926a20ee72a986e0763101fdf366f27c9cab5d` |

大型材料和工作区不进入 TorchPort Git。实验用的源仓库副本在
`migrations/CLIP/source/openai-CLIP`。该副本只有一个额外的 `torchport_adapter.py`。
适配文件的受控原件是本目录的 `source_adapter.py`。`materialize.py` 会检查两份文件相同。
候选仓库没有这个源端适配文件，也没有评分数据或参考 Jittor 实现。

## 固定任务

从 CIFAR-10 官方 `test_batch` 中，按类别顺序取每类最先出现的 10 张图，共 100 张。
预处理使用官方 CLIP `Resize`、`CenterCrop`、RGB、`ToTensor` 和 `Normalize`。
十个类别各有一个固定英文提示词。图像、token、标签、原始索引和三次源端 logits
保存在 `migrations/CLIP/data/pilot/`。`source_manifest.json` 记录每个文件的 SHA-256。

V100 上 PyTorch 2.4.1+cu121 的源端 top-1 为 `89/100`，三次推理的预测与 logits
逐位相同。源端适配入口加载 302/302 项权重。它与官方 `clip.load` 的分批推理输出
逐位相同。Harness 一次送入全部 100 张图时，源端与分批输出的最大绝对差为
`3.8146973e-05`，预测完全相同。

## 使用

先进入 `D:\科研\TorchPort`。下列命令使用当前项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe experiments/p2/clip-vit-b32-v1/materialize.py `
  --candidate D:\科研\CLIP-Benchmark-v1\workspaces\none `
  --output migrations/CLIP/outputs/task-forward-none.json

.\.venv\Scripts\python.exe experiments/p2/clip-vit-b32-v1/materialize.py `
  --candidate D:\科研\CLIP-Benchmark-v1\workspaces\full `
  --output migrations/CLIP/outputs/task-forward-full.json
```

两份任务文件已生成。迁移 Agent 分别在 `none` 和 `full` 工作区工作。
向两组提供相同的 [提示词](prompt.md)。控制器把 none 映射到 `diagnostic`，
把 full 映射到 `localized`。固定输入、权重、提示词和预算必须保持一致。

目标端实现完成后，先执行 Harness `verify`。将该次目标端原始输出 NPZ 复制到控制区。
下面的脚本读取其中的 `output` 张量并计算 100 张图的 top-1 与预测匹配率：

```powershell
.\.venv\Scripts\python.exe experiments/p2/clip-vit-b32-v1/score_task.py `
  --pilot migrations/CLIP/data/pilot `
  --target-logits <target-outputs.npz> `
  --output migrations/CLIP/outputs/target-task-score.json
```

只有在试验前冻结 `tau` 后，才传入 `--tau <value>` 并计算 `P_t`。
缺少 `tau` 时，脚本将 `P_t` 和 `task_success` 记为 `null`。

## 当前检验与边界

`migrations/CLIP/outputs/harness-seed-smoke-r2/` 保留原始 Harness 返回、远端
`verify` 报告、oracle JSON 和 NPZ。空 seed 的 `verify` 返回 `RUN_FAILED`。
原因是 `run.py` 的预期 `NotImplementedError`。这证明入口已经执行到目标实现，
不表示迁移通过。源 oracle 的 checkpoint 状态是完整的，302 项均加载。

本任务目前只覆盖推理 pilot。CLIP 在 Benchmark v1 中还要求微调训练检查。
正式 trial 前仍需完成目标端实现与正例校准、训练 GATE、逐层检查点、`tau` 冻结、
Agent 成本记录和多次重复。100 张图的结果是固定 pilot 样本结果，不是完整 CIFAR-10 指标。
