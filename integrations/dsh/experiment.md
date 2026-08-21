# DSH 实验协议

DSH 不是固定迁移脚本。正式实验中，DSH 面对新仓库选择工具参数和修复动作；
TorchPort 只提供静态事实、运行结果和数值证书。

每轮必须记录：

```json
{
  "mode": "dsh",
  "round": 1,
  "decision": "基于哪个证据采取什么动作",
  "candidate": "候选通用修复"
}
```

正式对比保持四种隔离方法：

| 方法 | 可用能力 |
|---|---|
| `official_jittor` | 官方静态转换器 |
| `codex` | 纯 Codex，不允许使用 TorchPort |
| `pure_dsh` | 纯 DSH，不允许使用 TorchPort |
| `dsh_torchport` | DSH 与本插件完整闭环 |

统一指标为入口恢复、目标可运行、FP32 对齐、Strict 对齐、首次分歧、修复轮数、
人工干预和总耗时。没有双端证书的方法不能计为数值通过。

当前已完成一项真实 DSH 端到端方法验证：PyTorch examples SuperRes 在第 8 个逻辑
修复轮达到 `verified`，FP32 通过，Strict 首个分歧为 `conv3`。此前失败轮和最终证据
均保留，详见 `experiments/stage3_method/trajectory.md`。其他方法的正式批量基线仍需按
`experiments/prompts/` 独立执行，不能用本次结果代替或编造。
