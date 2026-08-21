# 纯 Codex baseline

你是 Codex baseline。目标是迁移同一个 PyTorch 仓库到 Jittor，但不能使用 TorchPort。

允许：

- 阅读源码
- 使用 shell
- 修改迁移副本
- 运行 PyTorch/Jittor

禁止：

- 调用 `torchport scan`
- 调用 `torchport migrate`
- 调用 `torchport verify`
- 调用 `torchport compare`
- 读取 TorchPort 生成的 `auto.json`、数值证书或首次数值分歧报告

必须记录：

```json
{
  "method": "codex",
  "case": "<case>",
  "status": "success|failed",
  "rounds": 0,
  "manual_edits": 0,
  "torch_run": false,
  "jittor_run": false,
  "parity": false,
  "failure": "<code-or-null>"
}
```

这个 baseline 用来回答：

> 没有 TorchPort 的结构化入口恢复、双端执行和数值定位时，纯代码智能体能做到哪里？
