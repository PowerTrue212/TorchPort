# Stage 3：DSH + TorchPort 方法验证

日期：2026-08-20  
样本：PyTorch 官方 examples `super_resolution` 子仓库  
模型：`model.py:Net(2)`  
输入：`1×1×16×16`  
目标：原生 Jittor

## 端到端结果

```text
status       verified
passed       true
FP32         true
Strict       false
first        conv3
weights      8/8
tensors      10/10
```

FP32 是最终迁移验收；Strict 用更窄容差检查全部中间层，因此 `conv3` 的容差级分歧
被保留为诊断证据，但不否决最终输出通过。

## 失败—修复轨迹

| 逻辑轮 | 结构化结果 | DSH 决策或工程结论 |
|---:|---|---|
| 1 | `TORCH_RUN_FAILED` | 发现构造参数被字符串化，改为整数 JSON 数组 `[2]` |
| 2 | `JITTOR_RUN_FAILED` | 目标端 JIT 初始化段错误，判定为运行环境而非模型语义 |
| 3 | `JITTOR_RUN_FAILED` | 尝试隔离缓存/CUDA，证据表明变量名与根因判断不足 |
| 4 | `TORCH_RUN_FAILED` | 旧插件 schema 再次产生 `['2']`，推动 JSON schema 修复 |
| 5 | `PARITY_FAILED` | 段错误消失，但 `conv1` 首先分歧，分块写入假设不成立 |
| 6 | `PARITY_FAILED` | 权重键 8/8 匹配但实际值损坏，继续检查数组桥接 |
| 7 | `JITTOR_RUN_FAILED` | 确认 NumPy 2/Jittor ABI 冲突；又发现运行检查未用 targetPython |
| 8 | `verified` | 统一目标解释器为 NumPy 1.26 环境，FP32 通过后停止 |

## 保留的通用改进

- DSH `modelArgs`、`modelKw` 和输入改为真实 JSON schema。
- `torchport auto` 的运行检查与采集统一使用 `targetPython`。
- 目标端执行前检查 Jittor/NumPy 数组桥接，防止“权重键匹配但值已损坏”。
- 子进程返回 0 但未生成 NPZ 时保留 stdout/stderr 并明确失败。
- 失败的分块权重写入实验未保留在最终代码中。

## 人工介入说明

DSH 完成了工具选择、参数修复、失败分类、重试和最终报告；开发阶段人工介入用于修复
插件 JSON 合同与 TorchPort 运行器本身。该样本证明插件闭环可运行，但不能替代后续
`official_jittor`、纯 Codex、纯 DSH 和完整方法在同一批未见仓库上的正式对比实验。
