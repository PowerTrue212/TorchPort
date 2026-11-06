# RAFT 全量 r5 试验：事故与进展报告

## 结论

本次 r5 全量试验属于**受基础设施影响的 pilot**，既不能判定为成功，也不能判定为干净的模型失败。DSH 多次收到 `Error: fetch failed`，但控制器仍写入了验证报告；用户最终在 step 92 因长时间等待停止了运行容器。隐藏的最终 verify 仍未完成。因此，**不能把 q2/q3 的 PASS 当作最终结果**。

## 控制器已确认的记录

| 查询 | 候选 | 结果 |
|---|---|---|
| q1 | 同一候选 | inference PASS，training FAIL |
| q2 | 同一候选 | inference PASS，training PASS |
| q3 | 同一候选 | inference PASS，training PASS |
| q4 | 候选变更后 | inference PASS，training FAIL |
| q5 | 已启动 | 停止前没有完成的工具结果 |

另：第一次 trace 调用发生在 verify 之前，正确返回“请先运行 verify”，trace 查询数为 0，因此没有取得 trace 数据。

## 本次会话的代码进展

本次会话已完成 `evaluation_entry.py` 的原生 Jittor 实现，主要包括：

- 标准非 small RAFT 架构，参数/缓冲区命名与上游 checkpoint 对齐。
- 使用显式 im2col + matmul 实现卷积，并设置 `stop_fuse`，避免触发目标 GPU 上 Jittor 原生卷积反向的 `cudaErrorIllegalAddress`。
- 自定义 PyTorch 兼容 BatchNorm2d，包含 running mean/var 更新；因 Jittor 无 0 维 Var，省略标量 `num_batches_tracked`。
- 为共享的 `norm3` / `downsample.1` 补充 state_dict 别名；兼容 checkpoint 的 `module.` 前缀。
- 实现 20 次迭代推理、12 次迭代训练、upstream sequence loss（`gamma=0.8`、`max_flow=400`）。
- 实现 AdamW（lr `2e-5`、wd `5e-5`、eps `1e-8`）、103 步 OneCycle（`pct_start=0.05`、linear）、梯度裁剪 `1.0`，并暴露 `param_groups` 与当前学习率。
- 后续修改包括仅将可训练参数交给优化器，以及兼容 DataParallel checkpoint 前缀。会话内还做过本地 CPU 冒烟测试及与上游 PyTorch 的数值对照，但这些不能替代远端最终验证。

以上是代码侧事实，**不等同于本轮远端验证结论**。特别是 q4 的 training FAIL 发生在候选变更后，但由于 DSH 未收到报告正文，无法判断它是候选回归、BatchNorm 训练语义差异，还是基础设施/验证波动。

## 剩余未知

1. 隐藏的最终 verify 是否完成、结果如何。
2. q1/q4 training FAIL 的具体失败指标：loss、逐 step 参数差异、哪类参数或哪个 step 开始偏离。
3. q2/q3 PASS 是否可重复，是否会受候选变更和远端噪声影响。
4. training 失败是否与 BatchNorm 是否冻结、共享别名参数、优化器状态或 MCP 传输有关。

## 建议下一步

1. **不要将 q2/q3 宣称为最终成功**，等待控制器或隐藏最终 verify 的正式结果。
2. 先确认 MCP/verify 传输稳定，再对当前候选执行一次 verify/trace/localize，确保工具结果能实际返回。
3. 若 training 仍 FAIL，优先收集逐 step loss 与参数 diff，重点排查 BatchNorm 训练模式（是否应 `freeze_bn`）、`norm3`/`downsample.1` 别名更新，以及 AdamW/OneCycle 状态是否同步。
4. 在基础设施恢复前，将本次 r5 记录为“受传输故障影响、最终结论待定”的 pilot，而不是有效通过或确定失败。
