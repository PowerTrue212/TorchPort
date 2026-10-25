# LoRA GPT-2 Small：V100 训练复测（2026-10-24）

这是 pilot 校准，不是 Agent trial。主机为 Tesla V100-SXM2-16GB，驱动 535.183.01；
source 为 PyTorch 2.4.1+cu121，target 为 Jittor 1.3.11.0 / CUDA 11.3。
本次 TaskSpec 与 checkpoint 未改，V100 独立重建了 source oracle；不能复用或覆盖
RTX 4090 的 oracle。

正确的人工校准 target 先做一次 `verify`，source oracle 两次捕获一致，target 两次
均通过。噪声包络最大 GATE 比值为 3.86 和 3.15。原始报告：
[verify JSON](remote-v100-20261024/0001-verify.json)。然后在同一 trial/cache 下做
两次 target `run` + `evaluate`，均通过，最大 GATE 比值为 2.43 和 2.00。
原始 [第 3 次 run](remote-v100-20261024/0002-run.json)、
[第 3 次 evaluate](remote-v100-20261024/0003-evaluate.json)、
[第 4 次 run](remote-v100-20261024/0004-run.json)、
[第 4 次 evaluate](remote-v100-20261024/0005-evaluate.json) 和四份 target 输出 NPZ
均保留。

四次 target 的首步 loss 为 14.32946491241455、14.329466819763184、
14.329465866088867、14.329465866088867，跨度 1.90735e-6。输出不是逐位稳定；
但这四次均在当前冻结包络内。与 [4090 校准](4090-calibration.md) 的两败两过
相比，V100 的初步重复性更好；四次通过仍不足以证明任意长期运行都必过。

## 训练负对照

两个负对照复制校准 target，除各自 `torchport_train.py` 中的一处注入外，其余
30 个文件内容一致；TaskSpec 的 source、输入、权重和训练协议不变。
`loss-sum` 将已按 mask 求均值的 loss 乘以 `mask.sum()`，模拟把 mean 改成 sum。
V100 运行时输入、checkpoint、权重、结构均通过，首步 target loss 为
415.5544738769531（正确 target 约 14.32946），数值 GATE 在 `output.loss.0`
首次失败，100 个 GATE 张量中有 76 个超限。原始
[verify JSON](remote-v100-negctl-loss-sum-20261024/0001-verify.json) 与 target 输出 NPZ
保留。该负对照证明当前 GATE 能拦下此类 loss reduction 错误。

`schedule-shift` 将调度器初始步数从 0 改为 1。source 四步学习率为
`[0, 1e-4, 2e-4, 1.5e-4]`，target 为 `[1e-4, 2e-4, 1.5e-4, 1e-4]`。
输入、checkpoint、权重和结构仍通过；精确检查在 `train.lr` 首次失败。
原始 [verify JSON](remote-v100-negctl-schedule-shift-20261024/0001-verify.json)、
[source oracle 元数据](remote-v100-negctl-schedule-shift-20261024/oracle.json) 与
[target 运行清单](remote-v100-negctl-schedule-shift-20261024/target-manifest.json) 保留。
该负对照证明学习率调度不依赖数值容差即可检出。

`adam-l2`（把解耦权重衰减错误地加进梯度）与正确 target 只有这一处改动。
清理已结束 trial 的 Jittor 编译缓存后，在同一 V100 上运行独立 `verify`：
输入、checkpoint、权重和结构均通过；数值 GATE 在首个更新后的
`output.param.1.transformer.h.0.attn.c_attn.lora_A` 失败，100 个 GATE 张量中
74 个超限，最大比值 8503.18。原始 [verify JSON](remote-v100-negctl-adam-l2-20261024/0001-verify.json)、
[target 输出 NPZ](remote-v100-negctl-adam-l2-20261024/target-outputs.npz)、
[oracle 元数据](remote-v100-negctl-adam-l2-20261024/oracle.json) 和
[target 运行清单](remote-v100-negctl-adam-l2-20261024/target-manifest.json) 保留。
该负对照证明当前四步 GATE 能拦下这类 AdamW/Adam+L2 语义差异。

## Pilot 中间数据保留边界

- 保留 TaskSpec、候选代码、冻结输入与 checkpoint 的内容身份；正式 trial 另需
  `trial.json`、`events.jsonl`、完整候选快照和最终验收原始报告。
- 保留可复算本次判断的 oracle（`oracle.json`、`inputs.npz`、`state.npz`、
  `outputs.npz`、`train_fp64.npz`、`train_alt.npz`）、target 的 `outputs.npz`、
  run manifest/capture 与 `verify` 原始 JSON。摘要表可从这些原始量重算。
- Jittor 编译缓存和同步包是可重建临时产物，运行结束且原始报告已保存后可删。
  同一次 oracle 的两份 source run `state.npz` 经逐字节比较相同后，用硬链接
  去重：两个路径和文件内容均保留，不再占两份物理空间。冻结 oracle 目录中的
  `state.npz` 保持独立，避免将来显式重建 oracle 时改动旧 run 证据。
- 不按文件名直接删除 oracle、失败运行、日志或候选修订；正式 trial 的无效尝试
  也要保留标记，不能因其失败而从统计中消失。

待完成：其余训练负对照、LoRA-only checkpoint 回环、完整 E2E 数据及任务指标、
Agent/控制端硬隔离与 URL 审计。不能将这次技术校准计入 none/full 正式结果。
