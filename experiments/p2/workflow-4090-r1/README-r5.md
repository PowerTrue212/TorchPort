# RAFT full r5：trace 写入保护与重跑

日期：2026-11-04。r3、r4 的 RAFT full 均为远端磁盘写满导致的基础设施失败；原始记录分别留在 `D:\科研\Workflow-4090-r1\status-r3.md` 和 `status-r4.md`，不改写为模型失败或成功。

## 协议与实现

- trial 内无成功返回的 verify 报告时，Gateway 的 trace 在派发前返回“请先运行 verify”，不计 oracle query。
- verify 有可聚焦位置时，只采集前 8 个张量对应模块。无可聚焦位置返回 `TRACE_NO_FOCUS`，不启动执行进程。
- source 与 target 每端在写 NPZ 前按数组原始字节数加 ZIP/元数据余量估算；超过 4 GiB 返回 `TRACE_TOO_LARGE`，不继续下一端。

## 4090 控制端体积探针

控制端使用冻结 RAFT 参考实现和已存在的 v3 verify E1 张量顺序。在独立 `trace-probe-r5` 目录中把真实 E1 的指定位置标为聚焦位置；这是体积探针，不是新的数值验收。

| 聚焦位置 | source | target | 结果 |
| --- | ---: | ---: | --- |
| `fnet.layer3.0.downsample.0`，position 46 | 268,532,007 B | 92,358,128 B | 双端完整运行 |
| `update_block#20.2`，position 604 | 预计 4,552,509,056 B；实际仅约 6 KB 控制文件 | 未启动 | `TRACE_TOO_LARGE` |

原始控制端报告和大小汇总保存在 `trace-probe-r5/`。探针使用独立复制的 TorchPort `src`，没有改动冻结 oracle bundle。

## r5 工作区

- Candidate：`D:\科研\RAFT-Benchmark-v1\workspaces-wf-r5\full`；种子提交 `2f8bcb53cfc476e43a0db83c18329320c585c266`，工作区干净。
- Controller：`D:\科研\RAFT-Benchmark-v1\controller\raft-wf-full-r5`；localized、15 query、14400 s、端口 18967、`pilot_only`。
- 推理 digest `efc4d43155b728b584c59b9bc37e8cb57502b2520bbd238eea4ab502d9fe1ef1`；训练 digest `5586ca6424603914791b15252cfd26782f93064a35494f954f07c2610d3999fd`，与 r4 一致。
- 提示词 `prompt-raft.md` 未改。新 Jittor 缓存 `jittor-cache-wf-r5-full` 用控制端参考预热成功（loss 12.936839，参数 5,260,416）。
- DSH r5 于 2026-11-04 20:58（北京时间）启动。结果以 controller 的 `trial/result.json`、`official-verify.json` 与原始事件/运行产物为准。

## 运行中断与 DSH 报告

用户于约 21:55 要求停止长时间等待的 DSH。控制端停止容器后 DSH 在 step 92 正常退出；
外层控制程序继续冻结候选并运行隐藏验收，原始证据保留。第一次 verify 前调用 trace 得到
“请先运行 verify”，trace query 为 0。四次完成的 verify 中，q1 推理通过/训练失败，
q2 和 q3 同一候选双任务通过，q4 修改后推理通过/训练失败；q5 已发起但未返回完整工具结果。
DSH 客户端对这些长调用收到 `fetch failed`，所以 r5 full 不能作为有效的反馈效果 trial。
保存的 DSH session 已另行恢复，只读生成 [DSH 报告](raft-full-r5-dsh-report.md)；报告与原始
`controller/raft-wf-full-r5/dsh-report.jsonl` 一同保留。正式隐藏验收单独记录，不回写上述判定。

## 隐藏验收收尾

控制程序随后完成候选仓库快照与隐藏 `verify`。`trial/result.json` 记录 `passed=false`、
`valid=false`、`claim_status=pilot_only`，5 次 logical query。官方训练通过临界加跑，
官方推理 `RUN_FAILED`：Jittor 在输出阶段无法再分配 56,623,104 字节 GPU 内存。
这是一份运行失败证据；停止 DSH 后是否有 q5 遗留进程占用显存，目前仅有推断，不能据此确认根因。
r5 因 DSH 未收到多次 verify 反馈，仍不计入有效的 full 组样本。
