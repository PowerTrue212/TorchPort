# TorchPort P2 harness 与 COD GPU 验收报告

日期：2026-10-04  
远端工作目录：`/root/autodl-tmp/tp-p2-final-20261003`  
GPU：NVIDIA GeForce RTX 3080 Ti

## 结论

P2 最小闭环已经完成。受控 Agent Gateway、反馈边界、Logical Query Cost、Physical
Evaluation Cost、按工具聚合、固定输入、public/hidden TaskSpec 和 CLIP/RAFT 物化器均已
实现并通过本地回归。远端合并报告为：

- `infrastructure_passed = true`：两个 task 都完成 CUDA probe，public/hidden 都产生可信
  evaluator 结果；
- `candidate_passed = false`：CLIP 历史候选通过，RAFT 历史候选得到真实数值失败；
- `formal_claims_allowed = false`：本次 worker 没有容器或 bubblewrap，属于
  `trusted_candidate_pilot`，不能据此发表反馈条件效果结论。

这次完成的是 harness 与 benchmark 基础设施验收。六个正式 trial 已冻结为空白、相互独立
的 candidate workspace，但没有把历史候选或本次验证结果冒充为
verdict/diagnostic/localized 的 Agent 对照实验。

## 完成内容

### 受控 probe

`experiments/harness/gateway.py` 增加 `probe(path, args=[])`：

- 只执行 target workspace 内已有的 `.py`；
- 使用 TaskSpec 冻结的 target Python、device、env、timeout、cwd 和 trial cache；
- argv 不经过 shell；
- stdout/stderr 完整落盘，响应尾部限制为 20,000 字符；
- timeout、失败和成功均计入 Physical Evaluation Cost；
- 不增加 evaluate/verify/trace 的 Logical Query Cost，也不能签发成功。

`src/torchport/execution.py` 在继承 controller 环境时清除 credential 类变量，再应用
TaskSpec 显式环境。测试覆盖路径越界、参数传递、凭证清除、超时和输出上限。

### 成本与候选轨迹

`experiments/harness/aggregate.py` 现在从 `events.jsonl` 确定性派生：

- `tool_calls_by_name`；
- `tool_time_seconds_by_name`；
- `physical_evaluation_cost_by_tool`；
- candidate revisions 与 unique candidate identities；
- 首次 run 成功、首次开发 verify 通过、official verify 完成时间。

Logical Query Cost 仍只计 `evaluate / verify / trace`；GPU seconds、进程墙钟和 executions
仍属于 Physical Evaluation Cost，二者没有合并。

### TaskSpec、capture 与验收修复

- 固定 `.npy` 输入进入 TaskSpec，运行时校验 shape/dtype，oracle identity 使用内容 SHA，
  不依赖机器绝对路径。
- checkpoint 支持 `.npz`，并处理 source/target wrapper 的 `model.` 前缀。
- FP32 比较接受 tuple/dict 展开产生的 `output.*`，不再错误要求唯一的 `output` 键。
- 单元素 scalar/singleton 状态可以按目标 shape 加载并回读核验。
- `attn_mask` 与 `num_batches_tracked` 一样作为确定性、非可移植状态排除；真实多余参数仍会
  触发 `WEIGHT_LOAD_FAILED`。
- Jittor CLIP adapter 使用 `ModuleList` 注册 Transformer blocks，使 302 个权重都进入状态
  核验，而不是只验证顶层参数。

### Benchmark 物化

`experiments/p2/prepare_clip_raft.py` 为 CLIP/RAFT × verdict/diagnostic/localized 生成 6 个
独立 workspace，并分别生成 public/hidden TaskSpec。隐藏输入是 CLIP batch 反序与 RAFT
帧对交换，四个文件的 SHA256 写入冻结 manifest。

远端最终物化检查结果：6 个 target repo 路径互不相同；target Python 固定为
`/root/autodl-tmp/jt-env/bin/python`；Jittor 运行时固定 `python_config_path`、系统
`libstdc++`、`conv_opt=1` 与 `use_cutlass=0`。这些设置解决了该 worker 的 Conda
`libstdc++` 版本和缺少 cuDNN 8 的兼容问题，没有增加第二套执行器。

### 批量 experiment controller 与恢复

`experiments/harness/experiment.py` 已补齐正式批量入口：

- 在首次 provider 请求前验证 task × condition × replicate 完整矩阵、candidate workspace
  唯一性和冻结 artifact 哈希；
- 冻结 `run-plan.json` 与 `runtime-snapshot.json`，记录 controller/双端 Python、关键包版本、
  GPU/驱动和 TaskSpec identity；
- 每个 attempt 都从冻结 seed 复制新的 candidate，基础设施失败默认最多重试一次；
- 用原子更新的 `run-state.json` 恢复中断实验，已完成 trial 不重复运行；
- 聚合器按预注册 trial 选择首个有效 attempt，重试不增加 Success@Budget 分母；
- provider、controller 或 official evaluator 的异常统一标为 `infrastructure_error`，不混入
  有效方法失败。

可信 pilot 必须显式传 `--allow-trusted-pilot`。正式 manifest 还必须提供隔离 worker 生成的
attestation，证明网络和宿主文件系统隔离、candidate 是唯一可写挂载、source 与冻结输入
只读、隐藏材料未挂载给 Agent，并记录不可变 image digest。controller 会校验并冻结这份
证据，但 attestation 本身不会创建 sandbox；隔离仍由外部 worker 强制执行。

## 负例与性能回归

三项指定负例均存在并通过：

1. 修改 source 代码后 oracle identity 改变并重建；
2. 两个不同 target 具有不同 candidate identity，不能复用 candidate 产物；
3. target 多出参数时，即使输出数值一致也返回 `WEIGHT_LOAD_FAILED`。

`tree_hash` 使用 `os.walk` 原地裁剪 `.venv`、`.git`、缓存、输出和常见构建目录，不再先
`rglob` 整棵树。本地完整测试由此前约 70 秒恢复到本次的 4.62 秒。

## 远端 GPU 证据

精简原始证据位于
[`experiments/p2/clip-raft-pilot/remote-validation-20261004`](../experiments/p2/clip-raft-pilot/remote-validation-20261004/)。
下载归档 SHA256：

```text
8adfef309eae02392647c1820d844da9d0492cd8b8c1386ba86aa8bf0e28f229
```

### 设备和缓存

CLIP 与 RAFT 的 target `job.json` 都记录 `device = cuda`；对应 RunManifest 的
`capture.framework.device` 也为 `cuda`。probe 输出 `cuda = 1`，且 `JITTOR_HOME` 位于：

```text
gpu-validation-results-final/<task>/evidence/<task>/cache/<probe-trial>/probe-target/jittor
```

正式 verify 的 `capture.framework.cache_path` 同样位于各 task 的 validation output 下，
没有落入用户级共享 Jittor cache。

### 候选结果

| Task | Public | Hidden | 关键数值 |
| --- | --- | --- | --- |
| CLIP | 3/3 通过 | 3/3 通过 | 602,436 个比较元素，0 个超差；max abs `1.5259e-5` |
| RAFT | `PARITY_FAILED` | `PARITY_FAILED` | public：201,181 / 1,579,008 超差（12.7410%），max abs `0.0101671`；hidden：229,637 / 1,579,008（14.5431%），max abs `0.0140339` |

CLIP 的 public/hidden physical cost 各包含 6 次 CUDA 配置进程（3 source + 3 target）。RAFT
在第一次 target 数值失败后按协议 fail-fast，因此每组为 4 次进程（3 source oracle + 1
target）。首次空缓存 probe 分别记录约 92.82 秒和 92.95 秒，主要包含 Jittor 编译；这些
数字是 CUDA 配置进程墙钟，不等同于 GPU active time。

## 本地验证

```text
pytest: 76 passed in 4.91s
DSH plugin: 5 passed
wheel: torchport-0.38.3-py3-none-any.whl built successfully
```

最终 wheel SHA256：`32efe16ded4b1bf5eac38c6a5529ad87b40f3bbb514d06accb32646a9f4f24e7`。

Python lint、compile 检查与 JSON 证据一致性检查也通过。`python -m build` 在当前 checkout
会被根目录已有的 `build/` 名称遮蔽，因此最终使用隔离 PEP 517 路径
`py -m pip wheel . --no-deps` 验证发行构建。

## 可解释边界

本次结果证明：受控接口可以让 Agent 修改代码、运行 probe、执行 candidate、请求不同
反馈层级，并把逻辑查询与实际 GPU 执行分开计量；public/hidden evaluator 能区分“可运行”
与“数值通过”。

它没有证明 localized 比 diagnostic 或 verdict 更有效。要回答该研究问题，仍需在具备
文件系统和网络隔离的 worker 上，从六个冻结空白 workspace 启动真实 Agent trial，并按
预注册预算收集 Success@Budget、Time-to-Success、Logical Query Cost、Physical
Evaluation Cost 与 token。现有历史 CLIP/RAFT 结果只用于验收 adapter 和 evaluator，
不得并入正式效率统计。

## COD 迁移烟测

首个 COD 任务已经接入 [CamoFormer 官方 PyTorch 仓库](https://github.com/HVision-NKU/CamoFormer)，
固定 revision `5f924a91a13707b71ca1b986924504464ffd0638`。TaskSpec 使用完整 PVTv2-B4
CamoFormer forward、固定 `[1, 3, 128, 128]` float32 输入和五个输出，不把简化 toy 网络
冒充真实迁移。物化器生成 verdict/diagnostic/localized 三个空白且互不共享的 candidate
workspace；public/hidden 输入、prompt 和上游 revision 都进入冻结 manifest。

[第三方 Jittor 实现](https://github.com/yinbow/CamoFormer-Jittor) 只用于 Agent 不可见的校准，
固定 revision `be601810aa1e249440a5da281101bdfaae898cba`。它不能计作 Agent 迁移结果，也不能作为
未见 benchmark。远端 RTX 3080 Ti 上的校准结果如下：

| 项目 | 结果 |
| --- | --- |
| PyTorch source | CUDA 完整前向成功，五个输出，1,100 个可移植状态，10.4627 秒 |
| 权重加载 | 1,100 / 1,100 成功；missing/extra/value mismatch 均为 0 |
| FP32 verify | `PARITY_FAILED`；79,772 / 131,072 超差（60.8612%）；max abs 1560.2235 |
| Strict verify | 81,757 / 131,072 超差（62.3756%） |
| Verify physical cost | 1 次 CUDA-configured target execution，40.8662 秒 |
| Strict first divergence | `model.encoder.block1.0.attn.sr`，invocation 1，trace position 5 |
| 局部差异 | `[1,64,4,4]`；996 / 1,024 strict mismatch；max abs 0.00133434 |
| Trace physical cost | 2 次 CUDA-configured execution，321.7928 秒 |

该结果证明真实 COD 模型已经走通 TaskSpec、状态桥接、CUDA 执行、数值判定、trace 和
localize。它同时说明“已有 Jittor 版本”不等于严格数值对齐，因此 harness 没有把结构通过
误报为迁移成功。精简证据位于
[`experiments/p2/camoformer-pilot/remote-validation-20261004`](../experiments/p2/camoformer-pilot/remote-validation-20261004/README.md)；
1.4 GB 原始 tensor trace、下载仓库和 Jittor 编译缓存未纳入项目。

运行依赖已经由 manifest 的 `runtime_packages` 声明。原始数值校准快照保持不变；GPU worker
重启后又生成了最终 prepare-only 补充快照，记录 source 的 `timm 1.0.30`、`einops 0.8.2`、
`opencv-python-headless 4.14.0.94`，以及 target 的 `einops 0.8.2`。最终 snapshot SHA256 为
`1dedb67783c3f9ab20453beefe39d25c769a00ea148193949708c1ce0ef1e9e4`。重启后的驱动为
595.71.05，因此这份补充快照用于后续 Agent trial；旧驱动 580.105.08 的原始快照仍对应上表
中的数值校准。

下一步可以让真实 Agent 从三个空白 workspace 之一开始迁移。若要做正式条件对比，还需由
隔离 worker 产生 attestation，并按冻结 replicate 数运行三种反馈条件。更难的未见任务可用
[RUN](https://github.com/ChunmingHe/RUN)：它原始入口含多卡和仓库路径假设，物化时应先用
source adapter 把这些可复现性问题与框架迁移分开。

### V4.1 Flash 真实 Agent pilot

实际 pilot 已使用 API ID `deepseek-flash`；官方模型列表在运行日将该 ID 解析为
DeepSeek-V4.1-Flash。修正后的最终物化使用最小 target seed、200 行上限的 `read_file` 和每
trial 300k token。verdict/diagnostic/localized 均为有效失败，Success@Budget 为 `0/3`：三者
都在源码探索阶段耗尽 token，没有 apply_patch、candidate revision、run 或 verification。

因此本轮 Logical Query Cost 为 0，Agent 阶段 Physical Evaluation Cost 也为 0。隐藏官方
认证仍按协议执行，合计 12 次 CUDA-configured process、624.716 秒，均以空白 submission 的
`RUN_FAILED` 结束。这些官方执行不能算成 Agent verification 调用。

本轮不能解释 feedback ablation：三种条件都没有观察到 verdict、diagnostic 或 localized
响应。它证明 harness 能完整记录“在形成首个候选之前失败”，也暴露了 raw non-thinking
Chat Completions loop 在真实 COD port 上的适用边界。完整分析见
[`camoformer-v41-agent-pilot-report-2026-10-04.md`](camoformer-v41-agent-pilot-report-2026-10-04.md)。
