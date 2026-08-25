# RAFT PyTorch → Jittor 三方法对比（冻结版）

本目录完整沿用 CLIP 实验的组织与执行逻辑，而不只是复刻目录名：冻结配置、
三方法 prompt、统一输入与 reference、三份相同 workspace、公共 evaluator、Rounds
记录和最终报告都由各自文件负责。

```text
RAFT/
├── experiment.json
├── prompts/
│   ├── codex.md
│   ├── dsh.md
│   └── dsh_torchport.md
├── isolation/
│   └── dsh-pure.patch.yml
├── requirements/
│   ├── source.txt
│   ├── target.txt
│   └── viz.txt
├── scripts/
│   ├── prepare_fixture.py
│   ├── prepare_reference.py
│   ├── init_workspaces.py
│   ├── validate_protocol.py
│   ├── record_round.py
│   ├── evaluate_jittor.py
│   └── build_report.py
├── templates/submission.py
├── tests/test_protocol.py
├── source/princeton-vl-raft/
├── shared/
│   ├── checkpoints/raft-things.pth
│   ├── prepared/
│   │   ├── image1.npy
│   │   ├── image2.npy
│   │   ├── flow_low_ref.npy
│   │   ├── flow_up_ref.npy
│   │   └── weights_fp32.npz
│   └── provenance/
│       ├── bootstrap.json
│       └── reference.json
├── workspaces/{codex,dsh,dsh_torchport}/repo/
├── results/{codex,dsh,dsh_torchport}/
└── outputs/
```

## 冻结条件

第一轮固定为 `384 × 512`、`iters=12`、FP32、batch 1、
`test_mode=True`、`small=False`、`mixed_precision=False`、
`alternate_corr=False`。Fixture 使用官方 `frame_0016.png` 与
`frame_0017.png` 的相同中心裁剪，不缩放、不随机。

`shared/provenance/reference.json` 保存 V100 环境与所有正式 artifact 的
SHA256。主 NA reference 是 `flow_up_ref.npy`；`flow_low_ref.npy` 只作辅助记录。
Checkpoint 以 PyTorch 2.6 的 `weights_only=True` 加载，并导出三方法共用的中性
FP32 state archive。上游可选 `alt_cuda_corr` 不编译、不安装、不执行。

## 准备

```bash
cd /root/data-fs/RAFT
cp env.example.sh env.sh
bash bootstrap_env.sh
bash run_prepare.sh
```

`run_prepare.sh` 会生成/验证 fixture 和 PyTorch reference，然后从同一官方源码和
同一 `templates/submission.py` 初始化三份起点完全相同的 workspace。已存在且含
实验轨迹的 workspace 会拒绝覆盖。

## 三种迁移

分别使用 `prompts/codex.md`、`prompts/dsh.md` 和
`prompts/dsh_torchport.md`。Codex 与 pure DSH 明确禁止 TorchPort；只有第三种方法
可使用 TorchPort 诊断，但最终 RN/NA 仍只能来自公共 evaluator。

pure DSH 不能从普通的、已安装 TorchPort 插件的 DSH 会话直接启动。先运行隔离
启动器；它会在 DSH 最终配置层禁用 `torchport` 条目、清除子进程中的 TorchPort
环境变量，并将审计证据写入 `results/dsh/isolation/latest.json`：

```powershell
# 只做隔离审计
.\run_dsh_pure.ps1 -AuditOnly

# 审计通过后启动 pure DSH
.\run_dsh_pure.ps1
```

`prompt` 中的禁用条款仍作为实验协议，但不再承担技术隔离职责。安装清单里仍能
看到插件是正常的；有效判据是审计结果中的 `active_torchport_plugin=false`。

```bash
bash run_evaluate.sh codex
bash run_evaluate.sh dsh
bash run_evaluate.sh dsh_torchport
```

初次实现为 Rounds 0。公共 evaluator 失败后，每次连贯修复必须先通过
`scripts/record_round.py` 记录，最多 8 轮。

## 结果

Evaluator 统一计算 RN、NA、Max Error、Mean Error、Rounds，以及 Mean / Max /
P99 Framework Flow Difference。三种方法结束后运行：

```bash
bash run_report.sh
```

Agent 用时作为主要的成本侧辅助指标单独记录，不参与 `RN && NA` 成功判定。它从方法
接到任务开始，计到通过或人工停止的终态，并用 `agent_elapsed_kind` 区分平台提供的
`active_agent` 与会包含空闲/人工暂停的 `session_wall`。evaluator elapsed time 另列，
不能替代 agent 用时。agent turns、model/tool calls 和 provider 原生
input/output/reasoning/cache-read tokens 作为可选原始字段；cache-read 不并入 input，缺失值记为
`null`/`not_recorded` 而不是 0。由于不同 provider/model 的计数口径可能不同，token 只在
明确记录 provider、model 和 tokenizer 的前提下作辅助比较；agent wall time 也可能包含
空闲或人工暂停时间。

`scripts/build_report.py` 会在可用时读取各方法的 `agent_metrics.json`，生成
`outputs/agent_costs.csv` 与 `outputs/agent_costs.md`，并与正确性主表分开。

输出包括 `flows.png`、`errors.png`、非表格形式的 `method_results.png`、CSV、Markdown 和完整
`summary.json`。所有 flow 图使用同一个 optical-flow color wheel 与幅值尺度；三张
error map 使用同一 color scale。
