# 4090 工作流第二轮（r2）：清理远端，补跑 RAFT、LightGCN、FNO none

给 Windows 上的 DSH 执行者。第一轮见 `README.md`，规则与红线完全沿用，以下只写不同之处。
用户可能不在，**不要停下来等回复**：每步把命令、原样输出（最多 40 行）和 PASS/FAIL 追加到
`D:\科研\Workflow-4090-r1\status-r2.md`，失败按本步规定处理后继续。全部结束写一行
`WORKFLOW R2 DONE`。所有结果标 `pilot_only`。

工作目录 `D:\科研\TorchPort`；PowerShell 先设 `$env:TP_SSH_ENV_FILE = "$HOME\.tp_ssh3.env"`。
你写的 `.ps1` 必须存成 **UTF-8 with BOM**（Windows PowerShell 5.1 会把无 BOM 的中文路径读乱）。
新脚本放 `experiments\p2\workflow-4090-r1\scripts-r2\`。

## 已由 Claude 完成（只读，不要重做）

| 项 | 结果 | 证据 |
| --- | --- | --- |
| RAFT 训练校准 r2 | 参考复跑共 24 次，留一 0 误拒（最大 usage 0.539）；有效负对照 neg-gamma、neg-gru、neg-lr110、neg-beta2、neg-wd5e2 全部被拒；基线 `cal-r2/baseline-r2/target-repeat-baseline.npz`，SHA256 `2c8048de9b10d812258cc64116f357ac5102364d847169977a97ad7f05241e38`；用该基线 verify 参考两次通过（max_gating_ratio 2.80） | 远端 `/root/autodl-tmp/RAFT-Benchmark-v1/cal-r2/` |
| LightGCN 推理 | source 主路径改为稠密邻接乘（`torchport_inference_adapter_r3.py`），两次 forward 逐位相同；参考两次通过（max_ratio 1.159），负对照 ×1.001 被拒（22870） | 远端 `LightGCN-Benchmark-v1/inference-r1/task-r3.json`、`evidence/batch-inference-r5.log` |
| SimCSE npz 回收 | none-agent、none-official、full-official 三份 npz 已拉回并按 manifest 复核（含硬链接）全对 | 各 `remote-evidence-*\npz-reverify.json` |

注：r1 的 neg-wd（5e-5→5e-4）误差只有最差参考的 1.11 倍，在 k=8 下任何规则都分不开，属无效
对照，不计入门槛。这一点由用户裁定，你不要改。

## 阶段 A：确认 SimCSE full-agent 证据完整

`D:\科研\SimCSE-Benchmark-v1\controller\simcse-wf-full-r1\remote-evidence-smc-full-agent\`
下应有 `smc-full-agent-npz.tar`（约 14.5 GB）。

1. 若已有 `npz-verify.json`：运行
   `.\.venv\Scripts\python.exe D:\科研\Workflow-4090-r1\scripts-r2\reverify_npz_tar.py <该目录> smc-full-agent`，
   期望输出 `"ok": true`。
2. 若没有 `npz-verify.json`（拉取被中断）：把残缺 tar **改名**为 `smc-full-agent-npz.partial-<时间>.tar`
   （不要删），再运行
   `.\.venv\Scripts\python.exe D:\科研\Workflow-4090-r1\scripts-r2\pull_simcse_npz.py <该目录> /root/autodl-tmp/SimCSE-Benchmark-v1/harness-wf-r1/026ec4297fca smc-full-agent`，
   然后做第 1 步。
3. 另外三份确认 `npz-reverify.json` 均 `"ok": true`；FNO full 的
   `D:\科研\FNO-Benchmark-v1\controller\fno-wf-full-r1\remote-evidence-fno-full\verify.json` 中 `bad` 为空。

门槛：四份 SimCSE 都 `ok`，FNO full `bad` 为空。不满足：**跳过阶段 B**，在 r2 状态里记 FAIL，
后续 trial 只在磁盘允许时跑（见阶段 D 的磁盘规则）。

## 阶段 B：清理远端（用户已批准，只限下列路径）

阶段 A 通过后才做。删除前对每个路径 `du -sh` 记录，删除后 `df -h /root/autodl-tmp`：

```
/root/autodl-tmp/SimCSE-Benchmark-v1/harness-wf-r1/026ec4297fca
/root/autodl-tmp/SimCSE-Benchmark-v1/harness-wf-r1/35495e1d68f8
/root/autodl-tmp/SimCSE-Benchmark-v1/harness-wf-r1/5a3c4364d3d2
/root/autodl-tmp/SimCSE-Benchmark-v1/harness-wf-r1/fe3f7facf59f
/root/autodl-tmp/FNO-Benchmark-v1/harness-wf-r1/1c8b40308575
```

一个路径一条 `rm -rf`，写死上面的完整路径，不用通配符、变量或 `cd` 后相对删除。其他任何路径
都不许删。期望可用空间 ≥ 28G。

## 阶段 C：准备 RAFT 与 LightGCN 的 none/full（FNO none 已就绪，不动）

照 `scripts\wf_prepare.py`、`scripts\prepare_fno_trial.py` 的写法写新脚本；只建新目录，已存在就报错退出。
两组同一 seed 提交、同一提示词，只在 `feedback` 上不同（none=`diagnostic`，full=`localized`）。

### C1 RAFT（推理 + 训练同一 trial，两个隐藏任务）

- 工作区 `D:\科研\RAFT-Benchmark-v1\workspaces-wf-r1\{none,full}`：`upstream\` 解自
  `D:\科研\RAFT-Benchmark-v1\materials\raft-upstream-2888e15.tar.gz`（先核对 SHA256 为
  `be527f8676313236ceca2d39c93df8aedf34cd57a0e765126f7b833c305b7b55`，不符就记 FAIL 跳过 RAFT）；
  根目录放 `evaluation_entry.py`，四个函数 `load_inference_model`、`load_train_model`、
  `build_optimizer`、`train_step` 都只 `raise NotImplementedError`。不放 checkpoint、不放参考代码。
- 先把远端基线拉到本地新目录 `D:\科研\RAFT-Benchmark-v1\controller\cal-r2\target-repeat-baseline.npz`，核对 SHA256 = `2c8048de…`（上表全值）。
- 控制目录 `D:\科研\RAFT-Benchmark-v1\controller\raft-wf-{none,full}-r1`，端口 none 18951 / full 18952。
- `task-training.json`：以 `controller\train-three-r1\task.json` 为模板，source/inputs/weights/protocol 原样；
  `target` 填 `repo`=工作区、`entry`=`evaluation_entry.py`、`model`=`load_train_model`、`device`=`cuda`、
  `python`=`/root/autodl-tmp/envs/target/bin/python`，`env` 同第一轮 RAFT（`conv_opt=1`、`use_mkl=0`、
  `use_cutt=0`、`nvcc_path`、`python_config` 指向 `python3.9-config`、`PATH` 含 cuda-12.8），
  `JITTOR_HOME=/root/autodl-tmp/RAFT-Benchmark-v1/jittor-cache-wf-r1-{none,full}`；
  `protocol` 增加 `train_envelope="train-repeat-envelope-v1-pilot"`、`train_noise_baseline`=本地基线路径、
  `train_noise_baseline_sha256`=`2c8048de…`。`load_task` 后 digest **必须**等于
  `5586ca6424603914791b15252cfd26782f93064a35494f954f07c2610d3999fd`，否则记 FAIL 跳过 RAFT。
- `task-inference.json`：以 `controller\inference-r1\task.json` 为模板，source/inputs/weights 原样，
  target 同上但 `model`=`load_inference_model`，`protocol` 换成
  `{"profile":"noise-envelope-v3","repeats":2,"timeout":1800,"pass_k":8.0,"floor_k":8.0,"noise_floor":0.0,"variants":[{"name":"fp64","dtype":"float64"},{"name":"benchmark","cudnn_benchmark":true},{"name":"channels_last","channels_last":true}]}`。
- `mcp-trial.json`：`tasks` 与 `hidden_tasks` 都是 `{"inference": ..., "training": ...}`，
  `max_oracle_queries` 15，`max_wall_seconds` 14400，`remote.root`
  `/root/autodl-tmp/RAFT-Benchmark-v1/harness-wf-r1`，`home` `/root/autodl-tmp/RAFT-Benchmark-v1/home`，
  `extra_roots` `["D:\\科研\\RAFT-Benchmark-v1"]`。其余文件（token、dsh.patch、preflight、audit-rules、
  container-trial）照 `wf_prepare.write_controller`；审计禁止路径再加 RAFT 的 `controller`、`source`、
  `materials` 目录和另一组工作区。
- 提示词：`experiments\p2\workflow-4090-r1\prompt-raft.md`（两组都含卷积框架缺陷说明，不要改）。

### C2 LightGCN 推理

- 新建本地 source 副本 `D:\科研\LightGCN-Benchmark-v1\source\LightGCN-PyTorch-dense-r1`：复制
  `source\LightGCN-PyTorch-pilot-r2`（忽略 `__pycache__`），再放入
  `D:\科研\Workflow-4090-r1\scripts-r2\torchport_inference_adapter_r3.py`。原目录不动。
- 从远端 `LightGCN-Benchmark-v1/inference-r1/` 拉 `inputs/0-users.npy` 与 `weights.npz` 到本地新目录
  `D:\科研\LightGCN-Benchmark-v1\inference-wf-r2\`，两端 SHA256 必须一致。
- 工作区 `D:\科研\LightGCN-Benchmark-v1\workspaces-wf-r1\{none,full}`：`upstream\` 解自
  `D:\科研\LightGCN-Benchmark-v1\source-snapshot-947ca2b3.tar.gz`（SHA256 按
  `experiments\p2\lightgcn-v1\prepare_train_trial.py` 中的常量核对），根目录 `evaluation_entry.py`
  只含 `load_inference_model()` 抛 `NotImplementedError`。
- 控制目录 `controller\lightgcn-wf-{none,full}-r1`，端口 18953 / 18954；task 以远端 `task-r3.json` 为模板，
  source.repo 改为上面的 dense 副本、inputs/weights 改为本地路径，
  `JITTOR_HOME=/root/autodl-tmp/LightGCN-Benchmark-v1/jittor-cache-wf-r2-{none,full}`。
  可直接用 `wf_prepare.write_controller`（10 次 query、2 小时）。
- 提示词：`prompt-lightgcn-inference.md`。

### C3 预热与预检

- 预热 Jittor 缓存：RAFT 用远端 `controller-reference-r1`，LightGCN 用远端 `inference-r1/reference-r2`，
  照 `scripts\warm_caches_r2.sh`。缓存里只能有 `cpython-39` 扩展。
- 每组：工作区 `git status --porcelain` 为空、无隐藏文件；`run_container_trial.py --skip-agent` **不要**用
  （第一轮它会占掉控制目录）。

门槛：某仓库任一项不过，记 FAIL，跳过该仓库的 trial，继续下一个仓库。

## 阶段 D：trial（串行，不并行）

顺序：RAFT none → RAFT full → FNO none（控制目录 `fno-wf-none-r3`，已就绪）→ LightGCN none → LightGCN full。

每个 trial 前：远端无残留进程、显存约 1 MiB；可用空间 RAFT 需 ≥ 15G、其他 ≥ 8G，不够就记
`SKIP（磁盘）` 跳到下一个，不删除任何东西。启动命令同第一轮：

```powershell
.\.venv\Scripts\python.exe experiments\harness\agents\run_container_trial.py `
  --controller <控制目录> --prompt <提示词> --proxy http://127.0.0.1:7890 --memory 5g
```

写一个新的 `scripts-r2\run_trials_wf_r2.ps1`（UTF-8 with BOM），照 `scripts\run_trials_wf_r1.ps1`。
基础设施故障（容器、代理、MCP 在 Agent 第一次模型调用前就断）最多重试一次；Agent 已开跑的
trial 一律不重跑。每个 trial 结束后用 `scripts\pull_trial_evidence_json_wf.py` 拉回 JSON/日志和完整
manifest；**不要**删远端 trial 目录。

## 阶段 E：汇总

照第一轮阶段 5，把五个 trial 的数字填进 `status-r2.md` 的同一张表（`passed`、`failure_code`、
query、执行次数、墙钟、模型步数、超限张量数与 max ratio），再列预检/审计命中、`agent_exit`、
`cleanup_error`。只填数，不写结论。最后写 `WORKFLOW R2 DONE`。
