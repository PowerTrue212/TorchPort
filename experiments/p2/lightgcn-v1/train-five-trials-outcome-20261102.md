# LightGCN 五步训练 Agent trial（3080 Ti，2026-11-02，pilot）

两组从同一 seed commit `bc1ac846091983dcc23e2bbca942b1f833939121`、同一
`prompt-train-five.md` 和同一 TaskSpec（oracle digest `cbbc8f7c…`，
`train-repeat-envelope-v1-pilot`，基数 SHA256 `d88be0a6…`，`repeats=1`）出发，串行使用
RTX 3080 Ti。none 接收 `diagnostic`，full 接收 `localized`。两组都在 `python_config`
修复（`dd501ee`）之后运行，各用全新 Jittor 缓存，缓存中只有 `cpython-39` 扩展。
均为 `pilot_only`、`valid=false`，不能计入正式 Benchmark v1。

## 有效对照

| 项目 | none-five-r5 | full-five-r3 |
| --- | ---: | ---: |
| 容器预检、审计 | 通过，0 命中 | 通过，0 命中 |
| Agent 退出码、模型步骤 | 0、63 | 0、60 |
| Logical Query Cost | 3（evaluate 2、verify 1、trace 0） | 7（evaluate 4、verify 2、trace 1） |
| 唯一 candidate 查询数 | 2 | 3 |
| Physical Evaluation Cost | 5 次、107.3 秒 | 8 次、117.9 秒 |
| 其中隐藏验收 | 2 次、8.3 秒 | 2 次、8.7 秒 |
| Agent 墙钟 | 1395.9 秒 | 775.4 秒 |
| Harness 工具耗时 | 124.3 秒 | 167.6 秒 |
| N：未缓存输入 / 输出 / 缓存读 | 95,999 / 57,411 / 5,867,520 | 66,745 / 39,791 / 3,669,632 |
| N：provider total | 6,020,930 | 3,776,168 |
| K：Agent 工具调用 | 96 | 72 |
| 隐藏验收 | 通过，25/25 零超限，max ratio 1.830 | 通过，25/25 零超限，max ratio 1.830 |
| 固定容差 fp32 | 0/41,055 不匹配，max abs 1.41e-6 | 0/41,055 不匹配，max abs 4.32e-6 |

- 逻辑查询中有被拒的调用：none-r5 第一次 `evaluate` 在 `run` 之前调用，返回
  `ValueError`（没有可评估的 run）；full-r3 有两次 `evaluate` 在验证服务端因"候选产物与当前
  代码不符，需重新 run"被拒。full-r3 的 `trace` 报"没有支持 trace 的推理任务"：trace 只支持
  推理任务，本 trial 只有 `train-step`，所以必然失败；随后 `localize` 报"没有 trace"是连带结果。
  这是 Harness 的限制，不是 Agent 用错。被拒调用按当时规则计入 Logical Query Cost，影响查询
  效率比较，不影响验收。
- 每组只有一次有效运行，差异不能解释为反馈层级的稳定效应。

### 口径变更（2026-11-02，不改写上表）

trial 结束后协议改为：任务全是 `train-step` 时不向 Agent 暴露 `trace`、`localize`；执行前即可
判定必然失败的调用（工具不可用、参数不合法、没有可评估的 run）不计 query；到达验证服务后被拒
的仍计。见 [P2 评测协议](../../../docs/p2-evaluation-protocol.md)。上表保留当时的原始计数。
若按新口径重算，none-r5 为 2（evaluate 1、verify 1），full-r3 为 6（evaluate 4、verify 2），
但两组的 Agent 行为是在旧工具列表下发生的，重算值只作参考，不替代原始记录。

本对照中 full 组的 localized 条件实际只比 diagnostic 多出 evaluate/verify 反馈的首次分歧等
定位字段，trace/localize 未能提供任何信息；解读 none/full 差异时须考虑这一点。

## E_m 临时计分

常数沿用 SimCSE r7 的临时值：T0=3600 秒、N0=5,000,000、K0=100、F0=1，待更多仓库后调整。
T 取墙钟减 Harness 耗时（上界），N 取 provider total，F 暂记 0（中间版本未事后 verify）。

| | T/T0 | N/N0 | K/K0 | C | E_m |
| --- | ---: | ---: | ---: | ---: | ---: |
| none-five-r5 | 0.353 | 1.204 | 0.96 | 0.719 | 58.2 |
| full-five-r3 | 0.169 | 0.755 | 0.72 | 0.442 | 69.4 |

R_e 两组均为 1，s（pilot 规则）均为 1。Agent 候选尚未做 LastFM 推理指标（P_t）
和逐层 B_ℓ；现有任务指标重放只覆盖控制端 Jittor 参考，见
[任务指标记录](task-metrics-3080ti-20261102.md)。

## 无效记录（保留，不计入）

| trial | 原因 | 证据 |
| --- | --- | --- |
| none-five-r1 | 基础设施：远端执行器把空 `python_config` 替换为 `python3.12-config`，Jittor 编出 `cpython-312` 扩展，Python 3.9 导入时 `assert jit_utils.cc`；Agent 代码一次未执行。原 `result.json` 记 `agent_completed`/`passed=false`，未改写 | `trial/remote-raw/*` stderr；修复 `dd501ee` |
| none-five-r2 | 修复前运行；通过隐藏验收，但依赖缓存中一个来源不明的旧 `cpython-39` 扩展，环境条件与 r3/r5 不同 | 缓存 `jittor-cache-trial-none` 同时含 312 与 39 扩展 |
| none-five-r3 | 基础设施：`dsh: TRANSPORT: DeepSeek Messages transport failed`，Agent 退出码 1，31 步，无 result | `agent.err.log` |
| none-five-r4 | 基础设施：WSL VM 重启，`cleanup_error`，`stop_reason=infrastructure_error`，逻辑查询 0 | `container-runtime.json` |
| full-five-r1、r2 | 已准备未启动 | — |

原始记录在 `D:\科研\LightGCN-Benchmark-v1\controller\lightgcn-train-{none,full}-five-r*\`。
远端证据尚未拉回归档，远端文件未删除。
