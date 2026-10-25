# Microsoft LoRA GPT-2 Small：从 PyTorch 迁移到原生 Jittor

你在 Microsoft LoRA 官方仓库的独立 Git 工作区中工作。将 GPT-2 Small + LoRA
迁移到原生 Jittor，覆盖模型、参数冻结、前向、LoRA checkpoint 和训练更新。
先用平常的代码工具检查仓库、依赖、GPU、显存、数据与 checkpoint，再决定实现和调试步骤。

## 目标与入口

在 `run.py` 提供以下入口；主要实现应放在正常仓库模块中，不要把完整模型塞进入口文件：

- `load_model()`：返回原生 Jittor GPT-2 Small，12 layers、hidden 768、12 heads、
  vocab 50257、LoRA rank 4、alpha 32、dropout 0；LoRA 注入 fused QKV 的 Q 和 V。
- `mark_only_lora_as_trainable(model)`：冻结所有非 `lora_` 参数。
- `lora_state_dict(model)` 与 `load_lora_state_dict(model, state)`：只处理 LoRA 参数；
  缺键、多键和 shape 错误必须失败。
- `build_optimizer(model)`：返回 optimizer 和 scheduler。按原仓库 AdamW 语义，
  learning rate `2e-4`、weight decay `0.01`、betas `(0.9, 0.98)`、epsilon `1e-6`、
  `correct_bias=True`；线性 warmup 2 步、总步数 6 步。
- `train_step(model, state, input_ids, labels, mask)`：完成一步训练并返回 mean loss。
  步内执行 backward、optimizer step、清梯度和 scheduler step，不裁剪梯度。

`evaluation_entry.py` 是固定评测入口，保持其接口不变。评测检查固定 token batch 的 fp32 logits、
结构与参数完整性、可训练参数集合、连续四步的 loss、学习率及每步 LoRA 参数。
初始 base + LoRA 权重由评测方加载。交付应是当前仓库内可处理不同输入的原生 Jittor
实现。请自行测试前向、训练更新与 LoRA checkpoint 的保存加载。本轮只评固定批次的
迁移行为，不评完整 E2E 数据集的文本指标。

## 工作方式

你可以使用平常的仓库浏览、编辑、Shell、Python、Git、测试、网页检索和远程代码搜索。
在当前 Git 工作区完成交付，并使用本会话提供的验证工具迭代。`verify` 检查推理与
训练，两项都通过才算通过。若参考公开资料，结束时列出 URL、版本或提交号及使用范围。

## 预算

- Token 预算：{{BUDGET_TOKENS}}
- 最大轮数：{{MAX_TURNS}}
- 最长墙钟时间：{{WALL_MINUTES}} 分钟

结束时用中文说明修改、验证证据、参考来源和仍未知项。
