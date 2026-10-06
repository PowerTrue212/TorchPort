# P3：有证据的迁移问题库

## 目标

问题库保存可复现、可复验、适用边界明确的迁移经验，用于回答：已验证经验能否帮助
Agent 在未参与条目整理的新模型上少犯同类错误并更快完成迁移。

它不是报错问答集。条目必须把观测、解释、修复效果和未知项分开，不能把首次分歧、
Agent 推测或一次成功直接写成通用根因。

## 非目标

- 不实现自动根因诊断、反事实干预或自动修复。
- 不自动执行条目里的 command、patch 或脚本。
- 不引入向量数据库、embedding 服务、在线训练或新的运行器。
- 不把放宽容差、更换 evaluator、改变输入或减少比较输出记为修复。

## 最小条目

首版使用普通 JSON。字段名和允许值先冻结如下：

```json
{
  "schema_version": 1,
  "issue_id": "jittor-numpy2-array-bridge",
  "title": "Jittor 与 NumPy 2 数组桥接不兼容",
  "category": "environment",
  "status": "verified_on_case",
  "trigger": {
    "source_framework": "pytorch",
    "target_framework": "jittor",
    "source_version": null,
    "target_version": "1.3.11.0",
    "device": "cpu",
    "code_pattern": "NumPy array to Jittor Var",
    "configuration": "numpy>=2"
  },
  "symptom": {
    "stage": "runtime",
    "failure_code": "ARRAY_BRIDGE_INVALID",
    "summary": "array contents are corrupted after conversion"
  },
  "discrepancy": {
    "description": "observed behavior and confirmed boundary"
  },
  "reproduction": {
    "fixture": "compatibility/fixtures/jittor-numpy2-array-bridge",
    "task": null,
    "instructions": ["run the pinned failing environment", "compare the converted array"],
    "expected_failure": "ARRAY_BRIDGE_INVALID"
  },
  "repair": {
    "description": "use the verified compatible NumPy version in an isolated environment",
    "patch": null,
    "constraints": ["does not modify the system Python environment"]
  },
  "evidence": {
    "before": "path/to/failing/result.json",
    "after": "path/to/passing/result.json",
    "protocol_digest": "sha256:...",
    "verified": true
  },
  "applicability": {
    "known": ["jittor==1.3.11.0", "numpy>=2"],
    "unknown": ["newer Jittor releases", "CUDA path"]
  },
  "provenance": {
    "source": "real_migration",
    "case_id": "case-id"
  }
}
```

允许值：

| 字段 | 允许值 |
| --- | --- |
| `status` | `observed`, `verified_on_case`, `generalized` |
| `category` | `api_semantics`, `state_loading`, `input_pipeline`, `execution_model`, `numerical_backend`, `environment` |
| `symptom.stage` | `load`, `runtime`, `structural`, `numerical`, `performance` |
| `provenance.source` | `real_migration`, `synthetic_regression` |

`description` 可以记录已知解释，但必须与 `applicability.unknown` 一起保留证据边界。条目
只引用可审计产物，不内嵌大张量、完整日志或聊天记录。

`category` 描述问题出现在哪一层，不判断错误由框架还是 Agent 造成。具体 trial 中的错误
修改、推理过程和停止原因属于实验记录；如果它暴露了可复现的兼容性事实，问题库只保存
该事实及证据。

## 准入和状态提升

1. `observed`：真实失败或明确标注的 synthetic regression 可稳定复现，并保存不可混淆
   的失败证据。此状态不声称修复有效。
2. `verified_on_case`：在相同 TaskSpec 和 evaluator 协议下保留 before/after；after 必须由
   普通 `verify` 通过。只通过 trace、改变容差或改变测试数据不能提升状态。
3. `generalized`：同一问题族和修复在至少两个相互独立的真实案例上达到案例级验证，且
   记录未覆盖版本、设备和代码模式。重复运行同一仓库不算独立案例。
4. 若新证据否定条目，应降级状态或缩小适用范围，不能只追加解释来保留原结论。
5. 旧 `compatibility/rules.json` 是历史输入。缺少 before/after、协议摘要或未知项的规则
   不自动获得上述状态。

## held-out 评测边界

条目整理集与正式评测集按三个键同时隔离：仓库、代码修订和问题族。测试任务不能在
条目、fixture、prompt 示例或调试输出中出现；同一项目的改名副本不视为 held-out。

对照实验只改变 Agent 是否能读取已验证问题库。模型版本、系统 prompt、初始 candidate、
工具集合、query/token/wall-time 预算、硬件和 evaluator 均保持一致。报告：

- Success@Budget 和同类错误复发率；
- Time-to-Success、Logical Query Cost 与 Physical Evaluation Cost；
- provider token 成本；
- 正确检索并使用条目的比例；
- 因错误套用条目造成的负迁移率。

问题库命中本身不算成功，只有独立 `verify` 通过才算迁移成功。

## 最小实施顺序

1. 建立 JSON schema 校验，并从真实迁移中选择两个高价值、可复现的推理问题。
2. 为每个条目保存最小失败 fixture、修复和同协议 before/after 证据。
3. 实现按 category/framework/version/device 的确定性只读过滤。
4. 条目和 held-out 任务冻结后，再把只读检索暴露给受控 Gateway。
5. 先跑小规模 held-out pilot；只有字段过滤无法满足真实查询时才讨论更复杂检索。
