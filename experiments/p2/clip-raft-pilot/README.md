# CLIP / RAFT P2 pilot

这个 pilot 把旧 CLIP/RAFT 案例迁入当前受控 Gateway。旧结果只用于选择任务和验证适配
层，不进入新的 Success@Budget 或成本汇总。

先准备两套冻结资产，再为每个条件创建独立 candidate：

```bash
PYTHONPATH=src python -m experiments.p2.prepare_clip_assets --device cuda
PYTHONPATH=src python -m experiments.p2.prepare_clip_raft \
  experiments/p2/clip-raft-pilot/materialized \
  --source-python /root/miniconda3/bin/python \
  --target-python /root/autodl-tmp/jt-env/bin/python \
  --replicates 1
```

物化器生成：

- `experiment.json`：六个预注册 trial；
- `workspaces/<trial>/repo`：从同一 task seed 独立复制的 candidate；
- `tasks/<trial>/public.json` 与 `hidden.json`；
- `private/<task>/*.npy`：CLIP 反序 batch、RAFT 反向帧对的隐藏输入；
- 每个隐藏输入的 SHA256。

每个 trial 只通过 `experiments.harness.deepseek` 暴露的 Gateway 操作。正式汇总使用物化后
的 `experiment.json`，不得并入 `CLIP/outputs` 或 `RAFT/results` 中的历史数据。

当前 RTX 3080 Ti worker 没有 container、podman、docker 或 bubblewrap。`probe` 和 target
执行虽然不会收到 hidden path 或 provider credential，但仍与 controller 共享宿主文件系统，
所以 manifest 固定为 `trusted_candidate_pilot` 且 `formal_claims_allowed=false`。正式论文
实验必须换到能够把 candidate 进程限制在 target/source/cache 挂载内并关闭网络的 worker。

GPU 基础设施验收使用历史候选的专用副本，不能把候选复制回上述正式 workspace：

```bash
PYTHONPATH=src python -m experiments.p2.validate_clip_raft_gpu \
  experiments/p2/clip-raft-pilot/materialized-validation \
  experiments/p2/clip-raft-pilot/validation-output
```

验证报告分开保存 `infrastructure_passed` 与 `candidate_passed`。前者要求 CUDA probe 成功，
且 public/hidden 都产生非基础设施错误的正式判定；后者才要求两个任务数值通过。2026-10-03
的 RTX 3080 Ti 验收中，CLIP public/hidden 各 3/3 通过，RAFT 两者都稳定返回
`PARITY_FAILED`，因此基础设施通过而历史候选整体未通过。精简证据位于
`remote-validation-20261004/`，它验证 task/harness，不是三种反馈条件的 Agent 效果实验。
