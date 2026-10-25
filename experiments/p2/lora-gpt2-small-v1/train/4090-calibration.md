# LoRA GPT-2 Small：RTX 4090 校准记录（2026-10-22）

这是 pilot 诊断，不是正式 benchmark 结果。新主机是 RTX 4090（sm_89，驱动
535.216.03，CUDA 12.1）；source 为 Python 3.11.10 / PyTorch 2.4.0+cu121，target
为独立 venv 中的 Jittor 1.3.11.0。与旧 V100 不共用 oracle。任务使用冻结的四步
token batch 与 base+LoRA checkpoint；没有运行完整 E2E 数据集。

前向校准通过两次 target 重复，fp32 最大绝对误差为 9.16e-5，固定容差下无失败元素。
原始 [前向 verify JSON](remote-smoke-4090-r2/0001-verify.json) 和两份 target 输出 NPZ 保留。
这两次运行的输入逐位相同，但 logits 有 1,283,277 / 1,608,224 个元素不逐位相同，
两次 target 间最大绝对差为 1.3733e-4。这说明运行间波动已存在于模型前向，
不能把训练首步 loss 波动仅归因于 loss reduction 或 optimizer。

训练校准的第一轮 `verify` 失败：103 个张量的固定 fp32 容差下无失败元素，但
100 个噪声包络 GATE 张量中 `output.loss.0` 超限。source loss 为
14.32946491241455，target 为 14.329460144042969，source fp64 参考为
14.329465391222424；source 噪声为 4.7881e-7，target 到 fp64 的误差为
5.2472e-6，高于当前探索门槛 `8 × noise + 1e-7 = 3.9305e-6`。
原始 [失败 verify JSON](remote-train-4090/0001-verify.json) 及同目录四份 NPZ 保留。

不改代码、不改 TaskSpec 和 oracle 后复跑 `verify`，两次 target 均通过，最大
包络比值分别为 6.98 和 4.98；原始 [复跑 verify JSON](remote-train-4090-r2/0001-verify.json)
及两份 target 输出 NPZ 保留。再运行一次 `verify`，首个 target 又因 `output.loss.0`
失败，最大包络比值为 12.95；[第三轮 verify JSON](remote-train-4090-r3/0001-verify.json)
及 target 输出 NPZ 保留。四次 target 的首步 loss 分别为
14.329460144042969、14.329462051391602、14.329463005065918、
14.329459190368652，跨度为 3.8147e-6，彼此并非逐位一致。四次运行中两次失败、
两次通过；同一 task digest、candidate identity 和 oracle 均未变化。`output.loss.0`
在第一次参数更新之前产生，因此优先检查前向和 loss 数值路径；这还不是算子根因。
不能把失败抹去，也不能把通过当成已稳定校准。现阶段不据此认定模型实现错误，
也不把 `k=8` 临时调大。若要把 4090 用于正式 trial，需先研究 target 波动，完成训练
负对照和重复验收规则；任何新容差都要作为新协议冻结并重跑。

此外，LoRA-only checkpoint 保存/重载回环、完整 E2E 数据与任务指标 `P_t`、
Agent/控制端文件隔离和 URL 审计仍未完成。当前 none/full 候选只能用于 pilot。
