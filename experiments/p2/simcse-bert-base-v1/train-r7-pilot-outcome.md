# SimCSE 有监督训练 r7 pilot（2026-10-29）

none-r7 与 full-r7-retry2 从同一 seed 提交
`340b9713891b723ce8ec6f901049c618b5606b80`、同一 `prompt-train.md` 和
修订 b 的 `train-repeat-envelope-v1-pilot` TaskSpec 出发，串行使用 V100。
none 接收 `diagnostic`，full 接收 `localized`。两组均标 `pilot_only`；
`valid=false`，不能计入正式 Benchmark v1。

| 项目 | none-r7 | full-r7-retry2 |
| --- | ---: | ---: |
| 容器预检、审计 | 通过，0 命中 | 通过，0 命中 |
| Agent 退出码、模型步骤 | 0、63 | 0、44 |
| Logical Query Cost | 1（verify 1） | 4（evaluate 1、verify 3；trace 0） |
| 唯一 candidate 查询数 | 1 | 1 |
| Physical Evaluation Cost | 10 次、1049.3 秒 | 12 次、1366.6 秒 |
| 其中 Agent 阶段 | 6 次、669.1 秒 | 8 次、624.0 秒 |
| 其中隐藏验收 | 4 次、380.2 秒 | 4 次、742.6 秒 |
| Agent 墙钟 | 1948.2 秒 | 2837.8 秒 |
| 隐藏验收 | 两次重复均通过 | 两次重复均通过 |
| 首个超限 | 无 | 无 |

两组的隐藏验收均检查了 6,384 个受 pilot 包络判定的输出张量；
两次重复各自均零超限。参数加载完整、输出结构一致。固定严格数值比较仍有
少量元素不匹配，因此结论限于修订 b 的 pilot 包络。`result.json` 的
`passed=true`、`valid=false`、`success=false` 与此一致。

full-r7-retry2 比 none-r7 多 3 次逻辑查询、2 次物理执行，Agent 墙钟多
889.7 秒。它的 Agent 阶段物理执行时间反而少 45.1 秒。full 的隐藏验收
在新远端目录重建 oracle，所用 742.6 秒与 none 的 380.2 秒不宜直接解释成
反馈层级导致的性能差。两组各只有一次有效 pilot，不能由此估计反馈层级对
成功率或效率的稳定影响。

full-r7-retry2 中，一次 `run` 和部分 `verify` 的 MCP 回传出现 `fetch failed`。
控制器仍保留已发生的远端物理执行；一次 `trace` 调用没有记入逻辑查询，
应以控制器 `result.json` 的分工具计数为准。Agent 可见 `evaluate` 通过，
最终冻结候选的隐藏验收也独立通过。回传故障未改变这两项判定，但影响
查询效率比较。

两次无效 full 记录原样保留，不并入上表：

- `full-r7` 首次：远端磁盘写满，Harness 调用报 `No space left on device`，
  隐藏验收同步失败；无最终判定。
- `full-r7-retry1`：MCP 连接断开，逻辑查询 0；Agent 退出码 1，
  `stop_reason=infrastructure_error`。其冻结候选后来通过隐藏验收，
  但 Agent 没获得 localized 验证反馈，不能与 none-r7 作有效对照。

原始 `container-runtime.json`、`agent.jsonl`、`audit.json`、
`trial/result.json`、`trial/official-verify.json`、远端运行返回与冻结
候选快照分别保存在控制目录：

- `D:\科研\SimCSE-Benchmark-v1\controller\simcse-train-none-r7\`
- `D:\科研\SimCSE-Benchmark-v1\controller\simcse-train-full-r7\`
- `D:\科研\SimCSE-Benchmark-v1\controller\simcse-train-full-r7-retry1\`
- `D:\科研\SimCSE-Benchmark-v1\controller\simcse-train-full-r7-retry2\`

远端证据归档见 `controller/remote-archive-final/manifest.json` 和
`controller/remote-archive-retry2/manifest.json`。远端文件只在用户确认
删除范围后清理。

## Benchmark v1 指标的原始量（2026-10-30）

四个指标尚未计分。下表只是可从现有日志直接取出的原始量；E_m 的归一化常数
T0/N0/K0/F0 未冻结，P_t、B_ℓ 尚无输入。提取脚本为控制端
`cleanup-20261029/extract_raw_costs.py` 与 `probe_token_records.py`。

| 原始量 | none-r7 | full-r7-retry2 | 来源与限制 |
| --- | ---: | ---: | --- |
| R_e | 1 | 1 | 冻结候选完成 16 步训练与隐藏验收 |
| s（pilot 规则） | 1 | 1 | 修订 b 的 pilot 包络，不是正式协议 |
| Agent 墙钟 | 1948.2 秒 | 2837.8 秒 | `trial/result.json` |
| Harness 工具调用耗时 | 1200.6 秒 | 1975.4 秒 | `events.jsonl` 的 `duration_seconds` |
| 墙钟减 Harness 耗时 | 747.5 秒 | 862.5 秒 | T 的上界：仍含 Agent 自己的编译和本地测试，分段计时未实现 |
| 模型轮数 | 63 | 44 | DSH `assistant/message` 条数 |
| N：未缓存输入 / 输出 / 缓存读 | 93,920 / 46,973 / 5,326,464 | 51,289 / 27,378 / 2,299,904 | DSH 会话 `assistant/message.data.usage` 逐条求和 |
| N：provider total | 5,467,357 | 2,378,571 | 同上 `totalTokens` |
| K：Agent 工具调用 | 78（bash 59） | 52（bash 39） | `agent.jsonl` 的 `tool_call` |
| Harness 见到的 candidate 版本数 | 2 | 2 | 不是 F：F 需对每个版本事后各 verify 一次，未执行 |

会话日志里 `usage` 在消息与流式块中各出现一次，两者逐项相等，只计一次。
模型标识为 `deepseek-flash`（none 另有 3 条 `deepseek-v4-flash`，来自会话标题与搜索子请求）。
每组只有一次有效 pilot，上表差异不能解释为反馈层级的稳定效应。

## 推理 P_t（2026-10-30）

用推理 pilot 冻结的 STS-B TaskSpec（`task_id=simcse-unsup-bert-base-stsb-forward-v1`，
官方 unsup checkpoint，1,379 对，τ=0.005 于 2026-10-27 冻结）对两组冻结交付快照
的 `evaluation_entry.py` 各跑一次隐藏 `verify`，再用 `score_stsb.py` 打分。控制目录
`simcse-r7-stsb-{none,full}-r1`；`simcse-r7-stsb-{none,full}`（无 `-r1`）因本机到
V100 断网未完成，作无效记录保留。

| | none-r7 | full-r7-retry2 |
| --- | ---: | ---: |
| 推理 verify | `WEIGHT_LOAD_FAILED` | `WEIGHT_LOAD_FAILED` |
| 可运行 / 结构 / 数值 | 通过 / 通过 / 通过 | 通过 / 通过 / 通过 |
| 缺少的 checkpoint 项 | `embeddings.position_ids`（199/200） | `position_ids`、`pooler.dense.{weight,bias}`（197/200） |
| 值不一致的参数 | 0 | 0 |
| STS-B Spearman（source 0.76853） | 0.76853 | 0.76853 |
| P_t | 100 | 100 |

推理验收按冻结协议失败：权重不完整不能默认成功，此结果不改写。缺失项是可由形状推出的
buffer 和 `cls_before_pooler` 不使用的 pooler，数值检查也通过，故 P_t 仍可计算；P_t 与 s
分开报告。原因是训练 trial 只检查训练入口，Agent 改写 `load_model()` 时没有保留推理
checkpoint 的完整 state。若 Benchmark v1 要求同一仓库同时提供训练与推理，prompt 和隐藏
验收都需显式覆盖推理入口。

P_t 只证明前向正确，不覆盖 loss、反向、优化器和调度；16 步训练后的权重离初始化太近，
不适合做训练版 P_t（待用 source 训练曲线确定缩减预算后另行设计）。

## E_m 临时计分（常数未冻结）

临时常数 T0=3600 秒、N0=5,000,000、K0=100、F0=1，由用户指定为临时值，跑过更多仓库后
再调；原始量已保存，常数变更时直接重算。T 取上表"墙钟减 Harness 耗时"（上界），
N 取 provider total，F 暂记 0（中间版本未事后 verify）。

| | T/T0 | N/N0 | K/K0 | C | E_m | 若 F=1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none-r7 | 0.208 | 1.093 | 0.78 | 0.598 | 62.6 | 59.0 |
| full-r7-retry2 | 0.240 | 0.476 | 0.52 | 0.338 | 74.7 | 69.5 |

`C = 0.4·T/T0 + 0.4·N/N0 + 0.1·K/K0 + 0.1·F/F0`，`E = 100·s/(1+C)`，s 取训练 pilot 判定。
两组差距主要来自 Token 用量；单次 pilot 不能归因于反馈层级。

## B_ℓ 探索性计分（2026-10-30，exploratory_not_formal）

控制端诊断探针 `probe_layers.py` 只在 13 个检查点（`model.embeddings`、
`model.encoder.layer.0..11`）挂 hook，fp32 原样保存，不经 `verify`、不签发验收。
输入为推理 smoke 64 句（64×64 token），官方 unsup checkpoint；source 跑 fp32 与 fp64
各一次，每个候选跑两次。打分 `score_layers.py` 并列两种公式：

- 固定容差（协议现行）：`d_ℓ = max|t−s| / (1e-6 + 1e-5·|s|)`
- 噪声标定（协议待定修订）：`d_ℓ = max|t−fp64| / (8·max|s−fp64| + 1e-7)`

`B = 100/13 · Σ 1/max(1, d_ℓ)`；最终输出单独报告，不进 B。

| | none-r7 | full-r7-retry2 |
| --- | ---: | ---: |
| B_ℓ 固定容差 | 41.0 | 40.3 |
| B_ℓ 噪声标定 | 100 | 100 |
| 噪声标定最大 d_ℓ | 0.26（layer.11） | 0.28（layer.11） |
| 逐层最大绝对误差 | 1.4e-6 ~ 1.5e-5 | 9.5e-7 ~ 2.1e-5 |
| 两次运行 | 逐位相同 | 逐位相同 |

两组每层与 source 的差都和 source 自身 FP32 对 FP64 的距离同量级（噪声标定比值
0.09–0.28）。固定容差版只有 embeddings 一层 d<1，其余 12 层 d 在 1.2–7.0，
所以约 40 分。这说明固定 a、r 对 12 层 BERT 偏紧，正确实现也只能得 40 分左右，
不能区分实现好坏；噪声标定版更合理，但检查点和容差均为事后选定，不是预注册分数。
推理 smoke 上两个候选的 target 两次运行逐位相同（与训练路径的跨进程波动不同）。

full 缺少 pooler，但 13 个检查点不含 pooler，B_ℓ 不受影响。两个候选的推理权重
加载均不完整（见推理 P_t 一节），B_ℓ 在此前提下计算。原始 NPZ 与 SHA256 在远端
`/root/data-fs/SimCSE-Benchmark-v1/blayer/out/`（约 1.1 GB），打分 JSON 已拉回
`controller/simcse-r7-blayer/score-{none,full}.json` 并核对哈希。

## 四个指标汇总（pilot_only）

| 指标 | none-r7 | full-r7-retry2 | 性质 |
| --- | ---: | ---: | --- |
| s（训练 pilot 规则） | 1 | 1 | 修订 b 包络，非正式协议 |
| R_e | 1 | 1 | 16 步训练与隐藏验收均完成 |
| P_t（推理，STS-B） | 100 | 100 | 只覆盖前向；推理 verify 为 `WEIGHT_LOAD_FAILED` |
| B_ℓ（固定容差 / 噪声标定） | 41.0 / 100 | 40.3 / 100 | 探索性，事后选定检查点 |
| E_m（临时常数） | 62.6 | 74.7 | 常数未冻结；F 暂记 0 |

每组只有一次有效 pilot，表内差异不能解释为反馈层级的稳定效应。
