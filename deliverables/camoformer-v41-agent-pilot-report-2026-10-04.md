# CamoFormer × DeepSeek-V4.1-Flash Agent pilot 报告

日期：2026-10-04  
性质：`trusted_candidate_pilot`，replicate = 1，不支持反馈条件效果结论  
GPU：NVIDIA GeForce RTX 3080 Ti 12 GB，最终 worker driver 595.71.05

## 结论

TorchPort harness 已经在真实 COD 模型上完成从冻结任务、受控 Agent、预算停止到隐藏官方
认证的完整闭环。最终 pilot 的 Success@Budget 为 `0/3`。三个条件都没有产生 candidate
revision，也没有调用 run/evaluate/verify/trace；失败发生在第一次实现和反馈之前。

所以这次结果不能说明 verdict、diagnostic 或 localized 哪个更有效。它说明当前的
DeepSeek non-thinking Chat Completions Agent loop 在 300k cumulative token 内没有从源码探索
转入实现。继续增加 TorchPort 诊断信息不会解决这个阶段的问题。

## 冻结配置

- Source：[HVision-NKU/CamoFormer](https://github.com/HVision-NKU/CamoFormer)，revision
  `5f924a91a13707b71ca1b986924504464ffd0638`。
- API model ID：`deepseek-flash`；运行时模型列表解析名称为 `DeepSeek-V4.1-Flash`。官方
  [更新日志](https://api-docs.deepseek.com/updates/) 与
  [模型列表](https://api-docs.deepseek.com/api/list-models/) 使用同一映射。
- Provider mode：Chat Completions、thinking disabled、temperature 0、max output 8192。
- 每个条件预算：300,000 total tokens、80 model turns、10 logical verification queries、
  5,400 秒墙钟。
- Target seed：仅 `submission.py` 与 `torchport_adapter.py`；source 只读且不复制到 target。
- `read_file` 每次最多 200 行；prompt 限定完整 `CamoFormer(None)` PVT forward 与五个输出。
- Runtime snapshot SHA256：
  `6470513b10c77f632250443fa0c219831e1d7ab3bf981078651077e9e722dfae`。

## 最终结果

| Condition | Success | Stop reason | Total tokens | Cached input | Code tools | Logical queries | Candidate revisions | Agent executions | Official executions / seconds |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| verdict | 0 | token budget | 314,673 | 279,424 | inspect 1, list 2, read 20 | 0 | 0 | 0 | 4 / 207.185 |
| diagnostic | 0 | token budget | 312,731 | 283,008 | inspect 1, list 2, read 21, search 5 | 0 | 0 | 0 | 4 / 205.692 |
| localized | 0 | token budget | 327,646 | 299,264 | inspect 1, list 2, read 15, search 16 | 0 | 0 | 0 | 4 / 211.838 |

Aggregate：955,050 total tokens，其中 861,696 cached input、949,339 input、5,711 output；
56 次 read、21 次 search。Agent Physical Evaluation Cost 为 0。隐藏官方认证为 12 次
CUDA-configured process、624.716 秒。三份 official report 都返回 `RUN_FAILED`，直接原因是
空白 `submission.py` 仍抛出 `NotImplementedError`。

这里的 CUDA seconds 是 CUDA-configured 进程墙钟上界，不是 GPU active time。官方认证会
为每个 trial 构建三次 source oracle 并运行一次 target，因此它与 Agent 主动调用工具的成本
分开报告。

## pilot 中发现并修复的问题

第一版 target 错误地复制整份 PyTorch source，导致 Agent 从 source 与 target 重复读取同一
实现；现已改为最小 target seed。整文件读取也改为最多 200 行，并返回范围与是否截断。

另一个问题是 token 响应跨过预算时，adapter 在执行该响应的 tool calls 前退出。最终三个
边界响应的 `finish_reason` 都是 `tool_calls`。现在这些已经生成并计费的调用会先执行，预算只
阻止下一次模型请求。修复后的复跑确实保留了边界 read/search，但仍没有 apply_patch，说明
该 bug 不是 0 revision 的唯一原因。

## 解释边界与下一步

本轮没有进入 TorchPort feedback 阶段，所以不得把三个 0 分解释成反馈层级相同。replicate
只有 1，也不得做成功率显著性比较。已知 Jittor reference 只用于 hidden harness 校准，并未
暴露给 Agent；它加载全部 1,100 个状态但存在真实数值差异，证明 evaluator 并非只检查可运行。

下一次 Agent 试验应先更换 provider loop，而不是继续提高 token 上限。DeepSeek-V4.1-Flash
官方支持 Responses API 与 coding-agent 集成；可以在保持完全相同 TorchPort tools 的前提下，
改用 Responses/custom apply-patch 语义，或使用具有计划与上下文压缩的 DSH Agent。只有当
Agent 能产生首个 candidate 并调用 verification 后，verdict/diagnostic/localized ablation 才
开始回答原研究问题。

## 证据

- 最终 compact archive：
  `experiments/p2/camoformer-v41-final/remote-validation-20261004/agent-pilot-compact.tgz`
- archive SHA256：
  `35fd45d373214918162176d3b77358b63b2c1a245e32b62c630dffff93d2f85f`
- 展开后的 `summary.json`、`run-state.json`、三条 `events.jsonl`、三份 `result.json` 与三份
  `official-verify.json` 位于同目录的 `agent-pilot/`。
- V4.1 alias 快照：
  `experiments/p2/camoformer-v41-final/remote-validation-20261004/provider-model-snapshot.json`。

本地最终回归：pytest `76 passed`，DSH plugin `5 passed`，ruff 与 JSON/归档一致性检查通过。
