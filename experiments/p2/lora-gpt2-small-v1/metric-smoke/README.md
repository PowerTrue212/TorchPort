# Benchmark v1 四指标流程试跑

此目录只包含合成的 `P_t` 和 `E_m` 输入。它们测试公式和证据缺失处理，
不是 GPT-2 Small 的 E2E 任务结果，也不是 r1 Agent 成本。

试跑读取 r1 的冻结隐藏验收和逐层诊断：

- `R_e` 从每次隐藏运行的 `runnable` 字段计算。它不把数值验收失败误写为无法运行。
- `B_l` 从事后计算的逐层 JSON 读取。该值仍是探索性诊断。
- 不传入合成文件时，`P_t` 和 `E_m` 均为 `undefined`，数值为 `null`。
- 传入本目录的三个合成文件时，`P_t` 和 `E_m` 有演示数值。
  两项状态都是 `illustrative_synthetic`。报告固定为 `pilot_only`、`valid=false`。

入口为 `python -m experiments.harness.reporting.benchmark_v1`。它读取控制端原始文件，
不运行模型，也不签发正式验收。两份 r1 实际状态报告位于
`D:\科研\LoRA-Benchmark-v1\metrics-r1\pipeline-r1-current\`。
两份公式试跑报告位于
`D:\科研\LoRA-Benchmark-v1\metrics-r1\pipeline-smoke\`。
报告记录每份输入文件的 SHA256。

| r1 组别 | `R_e` | 探索性隔离 `B_l` | 当前 `P_t` / `E_m` | 合成公式试跑 `P_t` / `E_m` |
| --- | ---: | ---: | --- | ---: |
| full | 1 | 97.7499 | 未定义 / 未定义 | 100 / 66.2252 |
| none | 1 | 96.4341 | 未定义 / 未定义 | 100 / 66.2252 |

这份试跑只证明四个计算分支和缺证据分支可以执行。正式流程仍缺
GPT-2 Small E2E 任务权重、冻结评分程序、真实 Agent 成本事件和事先固定的
`B_l` 检查点与容差。不能用此试跑报告比较 none 与 full 的 `P_t` 或 `E_m`。
