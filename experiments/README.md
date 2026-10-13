# TorchPort 实验

本目录保存当前 P2 实验控制器、冻结配置、可追溯结果和数值验收图表。旧版 DSH 插件实验、
模型展示图和未按当前协议采集的历史结果不再保留。

| 路径 | 内容 |
| --- | --- |
| `harness/` | 成熟代码 Agent 的反馈控制、预算、事件、仓库冻结和结果聚合 |
| `p2/` | tiny smoke、CamoFormer 和 held-out 真实迁移实验 |
| `figures/fig_e1_ratio/` | E1 噪声标定比值 |
| `figures/fig_fixed_tolerance/` | 固定容差、运行波动和误差放大 |
| `figures/fig_jittor_atomic/` | Jittor 归约不确定性的机制证据 |
| `prompts/` | 当前架构与实验图示的提示词和来源说明 |

正式实验必须冻结 task、candidate 初始状态、模型版本、提示词、硬件、预算和停止条件。
原始 JSON、JSONL 与 NPZ 证据应保留在对应的 `p2/<study>/` 下；大型运行态文件继续由
`.gitignore` 排除，不得用缺失值或历史结果补齐当前统计。
