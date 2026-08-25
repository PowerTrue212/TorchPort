# TorchPort 实验

本目录用于把真实仓库验证转成可复现实验数据，而不是只记录“跑通了几个样本”。

## 实验数据分层

实验数据分为两层，避免把大体积运行态文件直接提交到 Git：

```text
V100 / data-fs（原始事实层）
├── source、checkpoint、fixture、reference
├── method workspace / submission
├── flow、source/target/weights NPZ
└── 完整 session log、trace 和 evaluator attempts

experiments/（可公开证据层）
├── results/<run-id>/
│   ├── manifest.json
│   ├── protocol.json
│   ├── summary.json
│   ├── methods/<method>/{result,rounds,termination,agent_metrics}.json
│   └── torchport/{certificate,compare_fp32,compare_strict}.json
└── figures/<study>/
    ├── data.csv
    ├── figure-spec.json
    ├── audit.json
    └── plot.png
```

`run-id` 使用不可变的“任务-协议-日期-硬件”形式，例如
`raft-v1-2026-08-25-v100`。`manifest.json` 只记录公开文件以及未提交原始文件的逻辑
路径、SHA256、字节数和保留位置；公开 JSON 中的绝对路径必须改写为相对路径或
`data-fs://...`，不得包含主机用户名、密钥或远端连接参数。`data/current.json` 是由这些
run bundle 派生的绘图索引，不作为原始事实来源。

正确性与成本分表：主表仍以 `RN && NA` 定义成功；成本表主要记录 agent 从任务开始到
通过/停止终态的 elapsed time，并以 `active_agent` / `session_wall` 标明计时口径；
evaluator elapsed 另列。agent turns、model/tool calls 和 provider 原生 token 字段作为
可选原始数据。缺失值
使用 `null`/`not_recorded`，cache-read token 单列，不与 input token 直接相加。不同
provider/model 的 token 口径不能直接解释为等价成本。

实验目标：

- 使用同一批 PyTorch 仓库，对比官方 Jittor 转换器、纯 Codex、纯 DSH、DSH + TorchPort。
- 每个方法都记录入口恢复、迁移、双端执行、权重加载、数值对齐、首次数值分歧、修复轮数。
- 所有图表由脚本从 JSON/CSV 生成，风格参考 `figures4papers` 的 `figure_*` 项目结构。
- DSH 决策必须写入 `agent` 字段；TorchPort 只提供确定性证据和工具输出。

当前状态：

- `data/current.json` 是 0.38.3 前收集的 8 个 TorchPort 双端证书快照；其中只有本次
  SuperRes 已由正式 DSH 插件端到端编排，其他样本不能用于证明 DSH 的增益。
- 已完成 8 个官方 Jittor 转换器样本，其中 4 个通过语法转换、4 个转换失败；该批量
  基线尚未统一执行 RN/NA/SA，不能把“未测”当作失败。
- 已额外补测同一个 SuperRes 样本：官方转换器通过 CV，但实例化时因
  `jittor.init.orthogonal_` 不存在而未通过 RN；DSH + TorchPort 通过 CV/RN/NA，SA
  在 `conv3` 出现首个严格分歧。
- 已从 8 个 TorchPort 双端证书快照提取注意力、特征、重建、通道权重、输出图和潜变量等
  可视化证据；其中只有 SuperRes 是正式 DSH 插件编排结果。
- `stage3_method/trajectory.md` 和 `results/dsh_superres*.json*` 保存本次真实 DSH
  SuperRes 闭环的失败轨迹、最终证书和工具调用证据。
- 其他 baseline 不能编造结果，按 `baselines.json`、`prompts/dsh.md` 和 `prompts/codex.md` 继续跑正式对比。

生成当前报告：

```bash
uv run --project . --extra viz python experiments/scripts/build.py
```

输出：

```text
experiments/out/cases.csv
experiments/out/methods.csv
experiments/out/case_table.md
experiments/out/method_table.md
experiments/out/model_table.md
experiments/out/metric_defs.md
experiments/out/errors.md
experiments/out/visuals.csv
experiments/out/visual_table.md
experiments/visual_plan.md
experiments/figures/fig_methods/plot.png
experiments/figures/fig_e2e/plot.png
experiments/figures/fig_rounds/plot.png
experiments/figures/fig_models/plot.png
experiments/figures/fig_vit_mae/plot.png
experiments/figures/fig_tensors/plot.png
experiments/figures/fig_vit_align/plot.png
experiments/figures/fig_mae_align/plot.png
experiments/figures/fig_fail/plot.png
experiments/figures/fig_status/plot.png
experiments/figures/fig_strict/plot.png
experiments/figures/fig_visual/plot.png
experiments/figures/fig_vit/plot.png
experiments/figures/fig_mae/plot.png
experiments/figures/fig_swin/plot.png
experiments/figures/fig_resnet/plot.png
experiments/figures/fig_senet/plot.png
experiments/figures/fig_unet/plot.png
experiments/figures/fig_superres/plot.png
experiments/figures/fig_vae/plot.png
```

其中 `fig_e2e` 是当前可用于主文的同仓库端到端对照，`fig_rounds` 展示真实 DSH 修复
轨迹。旧的 `fig_methods` 只汇总不同深度的历史快照，不能作为正式方法排名图。

单独重绘两张有效性图：

```bash
uv run --extra viz python experiments/figures/fig_e2e/plot.py
uv run --extra viz python experiments/figures/fig_rounds/plot.py
```

对应的事实表在 `out/e2e_table.md`，官方基线原始结果在
`results/official_superres.json`。架构图生成提示词在 `prompts/architecture.md`。

模型图统一采用两列配对布局：`PyTorch` / `Jittor` 放在图像上方，具体视觉类型放在下方；不绘制第三列差分图。每对图使用同一颜色范围，数值差异单独写入 `visual_table.md` 并汇总为折线图。

这些图验证的是：在相同输入和相同权重下，迁移前后的内部表示和输出被保留。当前证书使用仓库测试配置，不能替代预训练模型在真实数据集上的任务质量评测；后者应作为独立实验运行。

远端重新提取紧凑张量包：

```bash
python experiments/scripts/extract.py \
  --runs /data/L202500448/torchport/artifacts/real-runs \
  --out experiments/data/tensors/model_maps.npz
```

正式实验必须补齐：

```text
official_jittor
codex
pure_dsh
dsh_torchport
```

官方转换器 baseline 先使用独立 runner，不经过 TorchPort：

```bash
python experiments/scripts/official.py \
  --case swin \
  --source /path/to/repo \
  --entry models/swin_transformer.py \
  --output artifacts/baselines/official_jittor/swin
```

其中 `codex` 和 `pure_dsh` 是智能体 baseline，不允许调用 TorchPort 的结构化分析、
数值对齐和首次数值分歧定位工具。`dsh_torchport` 才允许调用 TorchPort。
