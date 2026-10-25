# LoRA GPT-2 Small r1 试跑结案（2026-10-24）

## 结论

full 和 none 的冻结仓库快照均通过 V100 隐藏验收：固定 batch 前向、连续四步训练各重复
两次，全部通过。两份控制端结果仍都是 `passed=true, valid=false, success=false`。
这是**代码技术验收通过**，不是有效的 none/full 对照实验，更不能据此比较反馈效果、
Agent 用时或查询效率。

| 候选 | 冻结快照内容摘要 | 隐藏前向 | 隐藏训练 | 结果文件 |
| --- | --- | --- | --- | --- |
| full | `f13deaa4c794b29ff68c4a3bbac136a00116ffacba0f015eb53d42a8ff28d7f1` | 2/2 | 2/2 | `D:\科研\LoRA-Benchmark-v1\controller\full-r1-pilot\trial\result.json` |
| none | `13c0b57d20de05a89792beff47220529dc4e4b64ce75337f5532d23abf11bb36` | 2/2 | 2/2 | `D:\科研\LoRA-Benchmark-v1\controller\none-r1-frozen-recert-v2\trial\result.json` |

none 的第一次冻结停在 `delivery/` 后，未产生最终报告。补验直接取这份快照作为输入，
没有再使用可变的候选工作区；再次冻结后的内容摘要与原快照完全相同。补验 TaskSpec
与原试跑逐字段相同，仅 `target.repo` 改为冻结快照路径。补验结果的
`agent_wall_seconds` 和 `oracle_queries=0` 只描述补验控制进程，**不能**代替原 Agent
会话的数据。原 none 会话的事件、调用和工具错误保存在
`D:\科研\LoRA-Benchmark-v1\controller\none-r1-open-tools\trial\`，其中记录了
1 次失败的 `verify` 查询；当时远端数据盘写满，原始错误证据已归档。

full 的 `agent_wall_seconds=6.14` 秒同样不是实际开发时间：full Agent 没有在这份
控制进程内完成开发。full 控制端 `oracle_queries=0` 不能和 none 原会话的查询次数比较。
两组的 `claim_status` 本来就冻结为 `pilot_only`；none 原 Agent 会话还因污染而中止。
不能把技术通过追认为正式成功。

## 为什么不能比较 none/full

开放工具设置保留了网页、远程代码搜索、Shell 和普通文件浏览，符合本轮试跑意图；
但控制端和先前 full 交付物仍位于同一台 Agent 可访问的机器。none Agent 的原始
`agent.jsonl` 明确记录了读取并对比 full 冻结仓库、查看 full 的 `official-verify.json`，
以及生成直接读取 oracle 文件的 `_probe_target.py`。这不是两组只在 Harness 反馈上
不同的实验。原始会话日志位于
`D:\科研\LoRA-Benchmark-v1\controller\none-r1-open-tools\agent.jsonl`；
该文件末尾可能因中断而不完整，不能据此推断所有外部访问都已记录。

若要获得可比较的正式数据，应从新的相同初始候选重新启动两组：保留通用 Agent
的网页和远程代码搜索能力，但不要让 Agent 的本地可见目录包含另一组的候选、
控制报告或隐藏验证材料；两组使用相同基础 prompt、普通工具、GPU、预算与停止条件，
并保存完整 Agent 会话和 Harness 原始证据。当前 r1 不能通过修改结果标签修复。

## 证据保留与范围

- full 原始报告：`D:\科研\LoRA-Benchmark-v1\controller\full-r1-pilot\trial\official-verify.json`、
  `remote-raw/`、`delivery/`。
- none 原会话：`D:\科研\LoRA-Benchmark-v1\controller\none-r1-open-tools\agent.jsonl`、
  `trial/events.jsonl`、`trial/tool-errors/`、`trial/delivery/`。
- none 冻结补验：`D:\科研\LoRA-Benchmark-v1\controller\none-r1-frozen-recert-v2\trial\official-verify.json`、
  `remote-raw/`、`delivery/`；两项各 2/2，通过的物理执行共 8 次。远端原始
  JSON/NPZ（不含可重建缓存和重复的 548 MB 权重 state）另归档为
  `D:\科研\LoRA-Benchmark-v1\archive\none-r1-frozen-recert-evidence-20261024.tar.gz`，
  SHA256 为 `1460c983d694aa06af46ac6c2c960acf895447f51fced5003008b139336cea1a`。
- 先前磁盘写满的非权重证据已归档为
  `D:\科研\LoRA-Benchmark-v1\archive\none-open-tools-first-error-20261024.tar.gz`；
  V100 校准非权重证据归档为
  `D:\科研\LoRA-Benchmark-v1\archive\calibration-evidence-small-20261024.tar.gz`。
  大型远端临时目录在核对归档摘要后清理，未清理本轮补验原始报告。

本轮补验结束后，经核对路径、任务 ID、进程状态及本地归档摘要，清理了远端
`/root/data-fs/TorchPort/fc58f90d9b12` 和 `1ba2c2984860` 两个可再生工作目录。
数据盘可用空间由约 3.9 GB 回到约 7.4 GB。原目录中的编译缓存不保留；如需逐字节
恢复整个远端工作目录，现有归档不足，须重新执行验证。

本轮只验收固定 batch 的前向、四步训练和相关权重加载行为。没有运行完整 E2E
数据集指标；现有 base 加合成 LoRA checkpoint 不是任务训练后的 checkpoint。
