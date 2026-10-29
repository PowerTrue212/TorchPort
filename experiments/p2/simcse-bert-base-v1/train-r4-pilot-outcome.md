# SimCSE 有监督训练 r4 pilot（2026-10-28）

两组在相同的 `340b9713891b723ce8ec6f901049c618b5606b80` 初始提交、
`prompt-train.md`、16 步源任务和 `5g` 容器内存限制下运行。none 收到
`diagnostic`，full 收到 `localized`。两者都标记为 `pilot_only`，**均未通过隐藏验收**；
`valid=false` 不能写作正式 Benchmark v1 结果。

| 项目 | none-r4 | full-r4 |
| --- | ---: | ---: |
| 容器预检、审计 | 通过，0 命中 | 通过，0 命中 |
| Agent 退出码 | 0 | 0 |
| 模型步骤 | 138 | 137 |
| Logical Query Cost | 2（evaluate 1、verify 1） | 10（达到预算上限） |
| 唯一 candidate 查询数 | 1 | 8 |
| Physical Evaluation Cost | 16 次，668.4 秒 | 15 次，1235.8 秒 |
| Agent 墙钟 | 1856.4 秒 | 4459.4 秒 |
| 隐藏验收 | `RUN_FAILED` | `PARITY_FAILED` |

none 的最终 `train_step` 返回 Python `float`，而 target 采集契约要求 Jittor
张量，故在采集 loss 时抛出 `AttributeError: 'float' object has no attribute
'numpy'`。这是一项目标端运行失败，尚未发生数值比较。

full 的目标端能运行，结构检查通过；首个超出当前噪声包络的是
`output.loss.0`。FP32 汇总为 21,685,600 个被比较元素中的 286 个不匹配，
最大绝对误差约 `5.55e-4`。**首次超限不是已确认根因**，也不能仅凭误差量级
把它判为无害。该候选按当前 pilot 协议失败。

原始 `container-runtime.json`、`agent.jsonl`、`audit.json`、`trial/result.json`、
`trial/official-verify.json`、远端运行返回与冻结候选快照分别保存在：

- `D:\科研\SimCSE-Benchmark-v1\controller\simcse-train-none-r4\`
- `D:\科研\SimCSE-Benchmark-v1\controller\simcse-train-full-r4\`

none 的原始 `result.json` 中 Git diff 标为不可用：交付脚本对忽略的
`.scratch/` 使用排除 pathspec 时，`git add` 报错。冻结快照和文件 manifest
仍完整。已在**不覆盖原始 result** 的前提下，核对当前 candidate 与冻结
`content_digest` 完全一致，另存补取的 status、diff 和摘要到
`none-r4/trial/delivery/git-evidence-posthoc/`。交付脚本已修复，加入了含
`.scratch/` 忽略规则的回归测试；这份补取证据应明确标作事后恢复。
