# 入口模式覆盖率

这里记录 TorchPort 静态入口恢复能力，不记录模型专用适配器。

每个条目对应一种真实 PyTorch 仓库常见入口模式。`status` 使用：

- `covered`：已有基准仓库和回归测试覆盖。
- `planned`：暂不继续写特例，等真实仓库压力测试证明它是高频失败原因后再实现。

批量实验时，`auto.json` 中的 `entry_cert` 和 `failure.code` 是统计入口恢复率与失败
阶段的主数据源。
