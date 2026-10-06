# DSH 历史实验与 P2 边界

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

以下四种方法属于既有历史实验：

| 方法 | 可用能力 |
|---|---|
| `official_jittor` | 官方静态转换器 |
| `codex` | 纯 Codex，不允许使用 TorchPort |
| `pure_dsh` | 纯 DSH，不允许使用 TorchPort |
| `dsh_torchport` | DSH 与本插件完整闭环 |

统一指标为入口恢复、目标可运行、FP32 对齐、Strict 对齐、首次分歧、修复轮数、
人工干预和总耗时。没有双端证书的方法不能计为数值通过。

新的 P2 不沿用这四组作为反馈 ablation。P2 使用同一个 TorchPort evaluator 和三个
冻结条件：`verdict`、`diagnostic`、`localized`。现有 DSH 插件会返回完整失败原因、
`next` 和首次分歧，反馈粒度相当于 `localized`，但它本身不实施 P2 的路径隔离和统一
成本计数。三组都必须通过 `experiments/harness/`；不能仅在 prompt 中要求 Agent 忽略
已经收到的信息。

当前已完成一项真实 DSH 端到端方法验证：PyTorch examples SuperRes 在第 8 个逻辑
修复轮达到 `verified`，FP32 通过，Strict 首个分歧为 `conv3`。此前失败轮和最终证据
均保留，详见 `experiments/stage3_method/trajectory.md`。其他方法的正式批量基线仍需按
`experiments/prompts/` 独立执行，不能用本次结果代替或编造。
