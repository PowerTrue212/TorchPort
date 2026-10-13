# 兼容性知识与迁移问题

`rules.json` 是早期兼容性规则清单，保留为 P3 问题库的历史输入。它记录了适用版本、
检测条件、修复动作和复验方法，但尚未统一关联同协议 before/after 运行、适用边界和未知
项，因此不能自动获得 `verified_on_case` 或 `generalized` 状态。

新条目遵循 [P3 有证据的迁移问题库](../docs/p3-verified-issues.md) 的 schema 和准入规则，
按可观察的问题层次分类，不判断失败应归因于框架还是 Agent。存储的命令或 patch 只
作为证据和建议，不由 TorchPort 自动执行。

当前早期规则来自 qx-slai3 上的 ResNet-18 与 ViT-Tiny 双端验证。
