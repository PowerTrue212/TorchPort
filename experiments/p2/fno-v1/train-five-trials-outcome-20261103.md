# FNO Darcy 五步训练 trial：none-r1 / full-r1（RTX 4090，2026-11-03）

两组从同一初始提交 `ae321db1bf0090fbf25b786bb6f2f9ca133fb68b`、同一 `prompt-train-five.md`、
同一 r3 source oracle（digest `f3d97351…`）和固定重复基数（`f57d320c…`）出发，由 Windows 上的
DSH 按 [runbook](runbook-train-five-r1.md) 串行执行。none 收到 `diagnostic`，full 收到 `localized`。
两组均为 `pilot_only`、`valid=false`，不计入正式 Benchmark v1。

本轮在 2026-11-02 的口径变更之后运行：任务只有 `train-step`，trace/localize 不向 Agent 暴露，
localized 与 diagnostic 只差 evaluate/verify 反馈中的定位字段；执行前即可判定失败的调用不计费。

## 结果

| 项目 | none-r1 | full-r1 |
| --- | ---: | ---: |
| 容器预检、审计 | 通过，0 命中 | 通过，0 命中 |
| Agent 退出码、模型步骤 | 0、109 | 0、93 |
| Logical Query Cost | 4（evaluate 3、verify 1） | 7（evaluate 6、verify 1） |
| 唯一 candidate 查询数 | 4 | 6 |
| Physical Evaluation Cost | 15 次、388.5 秒 | 17 次、438.1 秒 |
| 其中隐藏验收 | 4 次、14.0 秒 | 4 次、13.8 秒 |
| Agent 墙钟 | 2387.7 秒 | 2375.8 秒 |
| Harness 工具耗时 | 562.6 秒 | 681.2 秒 |
| N：provider total | 16,629,464 | 11,022,617 |
| N：未缓存输入 / 输出 | 143,869 / 109,403 | 110,486 / 89,219 |
| K：Agent 工具调用 | 128 | 110 |
| 隐藏验收 | 通过，2/2 次运行各 0/125 超限，max ratio 0.84 / 1.05 | 通过，2/2 次运行各 0/125 超限，max ratio 1.47 / 1.17 |
| 固定容差 fp32 | 5/16,170 元素不匹配，max abs 1.76e-4 | 5/16,170 元素不匹配，max abs 1.76e-4 |

固定容差的 5 个不匹配元素与控制端参考 r1 的形态相同（同为 `loss.4` 附近的薄包络位置），
pilot 包络判定为通过。

## 开发期验证序列

这是第一个两组 Agent 都在通过前先失败过的仓库：

| 序号 | none（diagnostic） | full（localized） |
| --- | --- | --- |
| 1 | evaluate → `RUN_FAILED` | evaluate → `WEIGHT_LOAD_FAILED` |
| 2 | evaluate → `WEIGHT_LOAD_FAILED` | evaluate → 服务端拒绝（候选产物与当前代码不符，需重新 run），计费 |
| 3 | evaluate → 通过 | evaluate → `STRUCTURE_MISMATCH` |
| 4 | verify → 通过 | evaluate → `STRUCTURE_MISMATCH` |
| 5 | | evaluate → `STRUCTURE_MISMATCH` |
| 6 | | evaluate → 通过 |
| 7 | | verify → 通过 |

- 两组都在开发期遇到过权重加载失败；full 另在结构检查上连续失败 3 次。是否由 localized 字段
  影响修复路径，本记录未逐条核对 Agent 看到的反馈内容。
- 两组 `agent.jsonl` 各有一次 MCP `fetch failed`，Agent 自行重试；控制端 `mcp.err.log` 为空，
  未影响最终判定。
- `audit.json` 的 `outside_workspace_paths` 含 `/data/...` 等引用和 full 的 docker 挂载参数文本，
  均未计为命中。

## 四个指标（临时）

| | s | R_e | P_t | B_ℓ（固定 / 噪声标定） | E_m |
| --- | ---: | ---: | ---: | ---: | ---: |
| none-r1 | 1 | 1 | 100 | 100 / 100 | 35.0 |
| full-r1 | 1 | 1 | 100 | 100 / 100 | 38.8 |

### E_m

临时常数沿用 SimCSE r7（T0=3600 秒、N0=5,000,000、K0=100、F0=1）。T 取墙钟减 Harness 耗时
（none 1825.1 秒、full 1694.6 秒，上界）；N 取 provider total。F 取开发期 evaluate/verify 判为失败的
candidate 版本数：none 2（RUN_FAILED、WEIGHT_LOAD_FAILED），full 4（WEIGHT_LOAD_FAILED、
STRUCTURE_MISMATCH×3）；full 第 7 次调用是验证服务内部崩溃，不计为候选失败。中间版本代码未被
Harness 保存，无法按协议事后统一 verify。

| | T/T0 | N/N0 | K/K0 | F/F0 | C | E_m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none-r1 | 0.507 | 3.326 | 1.28 | 2 | 1.861 | 35.0 |
| full-r1 | 0.471 | 2.205 | 1.10 | 4 | 1.580 | 38.8 |

### P_t 与 B_ℓ（RTX 4090 重放，2026-11-03）

source 训练 3 epoch 的 checkpoint（`c474de03…`）加载进两个冻结候选（24/24 参数，逐参数回读一致），
在 Darcy `darcy_test_16` 50 条上前向。source FP32 两次逐位一致，另跑一次 FP64。每个候选跑两次。

| | 平均相对 L2 | 输出与 source 最大差 | P_t | 检查点最大 d（固定 / 噪声标定） | 重复最大差 |
| --- | ---: | ---: | ---: | --- | ---: |
| source | 0.34482569 | — | — | — | 0 |
| none-r1 a / b | 0.34482569 | 4.8e-7 / 6.0e-7 | 100 | 0.545 / 0.149 | 1.2e-6 |
| full-r1 a / b | 0.34482569 | 5.4e-7 / 5.4e-7 | 100 | 0.593 / 0.158 | 1.2e-6 |
| 控制端参考 | 0.3448257 | 5.4e-7 | 100 | — | — |
| 负对照：输出 +0.001 | 0.3445557 | 1.0e-3 | **100** | — | — |

- B_ℓ 检查点为 `lifting`、`fno_blocks`、`projection` 三处输出，容差同 SimCSE/LightGCN（固定 a=1e-6、
  r=1e-5；噪声标定 k=8、floor=1e-7）。检查点和容差事后选定，标 exploratory。
- **P_t 漏检了输出 +0.001 负对照**：平均相对 L2 只变 2.7e-4，落在 τ 内。逐样本最大差 1e-3 很明显，
  但 P_t 只看汇总指标。这和第 7 步的观察一致：FNO 的 P_t 不能单独作为数值正确性的证据，需与 s 和
  逐样本差一起看。
- 两个候选的推理前向与 source 一致，推理指标区分不出 none 与 full。

每组只有一次有效运行，差异不能解释为反馈层级的稳定效应。

## 证据

`D:\科研\FNO-Benchmark-v1\controller\fno-train-{none,full}-five-r1\`：`container-runtime.json`、
`agent.jsonl`、`audit.json`、`trial/result.json`、`trial/official-verify.json`、`trial/events.jsonl`、
`trial/remote-raw/` 和冻结交付快照。

远端 `harness-train-five-r1/`（两组 trial 的开发期与隐藏验收工作目录）和 `replay-4090/`
（out、logs）已打包拉回 `D:\科研\FNO-Benchmark-v1\controller\fno-trial-evidence-20261103\`，
整包 SHA256 `14d5e242…`，解包后 1,790 个文件逐个复核 0 不一致。远端未清理。
