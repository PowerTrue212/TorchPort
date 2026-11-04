# 4090 无人值守工作流 r1（给 DSH 执行）

用户在睡觉，**不要停下来等用户**。任何一步失败：把原始输出追加到状态日志，按下面的
"失败处理"跳过依赖它的步骤，继续做其余步骤。全部做完后在状态日志末尾写一份汇总。

工作目录固定为 `D:\科研\TorchPort`，PowerShell 执行。所有结果标 `pilot_only`。

## 状态日志

`D:\科研\Workflow-4090-r1\status.md`（不存在就新建目录和文件）。每完成或失败一步，追加：

```
## <时间> <步骤编号> PASS|FAIL|SKIP
命令：...
关键输出（原样，最多 40 行）：...
```

新产生的控制端文件放 `D:\科研\Workflow-4090-r1\` 或各仓库的 `*-Benchmark-v1\controller\` 下
**新建**的目录；远端放 `/root/autodl-tmp/<仓库>-Benchmark-v1/` 下**新建**的目录。你新写的脚本
只放在 `experiments\p2\workflow-4090-r1\scripts\`。

## 红线（任何情况都不做）

- 不删除、不覆盖任何已有文件或目录（本地和远端都是）；同名就加 `-r2`、`-r3` 后缀。
- 不 `git commit` / `git push` / `git reset`；不改 `src/`、`experiments/harness/`、`tests/`。
- 不改已冻结的判定规则：`k=8`、`floor=1e-7`、`train-repeat-envelope-v1-pilot`、`noise-envelope-v3`
  的参数；不为了让参考通过而放宽门槛。门槛不过就是 FAIL。
- 不打印、不写入文件 `~/.tp_ssh3.env` 里的主机、端口、密码；不打印 `DEEPSEEK_API_KEY`。
- 隐藏 task、oracle、参考实现、基线文件不能出现在任何 `workspaces-*` 目录里。
- 远端不做全量 trace（曾写出 19 GB）。每个远端长任务前 `df -h /root/autodl-tmp`，可用 < 8G 就
  停止所有远端步骤，记 FAIL，跳到汇总。
- GPU 只有一张，所有远端任务和 trial 串行，不并行。

## 失败处理

- 准备阶段某仓库的门槛 FAIL：该仓库的 trial 记 SKIP，继续下一个仓库。
- trial 结果是 `PARITY_FAILED`、`RUN_FAILED` 等：这是实验结果，不是故障，照常记录，继续。
- 基础设施故障（Agent 还没开始就因 docker/WSL/MCP/代理/SSH 失败）：等 5 分钟，用 `-r2`
  新控制目录重试**一次**；再失败就 SKIP。Agent 已经开始运行的 trial 一律不重跑。
- 远端命令超时：用 `nohup` 后台跑，轮询日志文件（每 5 分钟一次），不要用阻塞命令等 1 小时。

## 连接与环境（只读背景）

```powershell
$env:TP_SSH_ENV_FILE = "$HOME\.tp_ssh3.env"
.\.venv\Scripts\python.exe scripts\remote\ssh_exec.py run "<命令>"
.\.venv\Scripts\python.exe scripts\remote\ssh_exec.py put <本地> <远端>
.\.venv\Scripts\python.exe scripts\remote\ssh_exec.py get <远端> <本地>
```

- 远端命令里不要内联复杂引号或 Python：写成本地脚本，`put` 上去再执行（PowerShell 引号会坏）。
- 远端没有 `python3`，用 `/root/autodl-tmp/envs/source/bin/python`（PyTorch）和
  `/root/autodl-tmp/envs/target/bin/python`（Jittor 1.3.11）。
- Jittor 任务环境：`conv_opt=1 use_mkl=0 use_cutt=0`，
  `python_config_path=/root/autodl-tmp/envs/target/bin/python3.9-config`，
  `nvcc_path=/usr/local/cuda-12.8/bin/nvcc`，PATH 前置 `/usr/local/cuda-12.8/bin`，
  每个任务/条件独立 `JITTOR_HOME`。缓存里出现 `cpython-312` 扩展就记 FAIL。
- **已知框架缺陷**：Jittor 1.3.11 在 4090 上原生卷积反向会报 `cudaErrorIllegalAddress`。RAFT
  控制端参考已把全部卷积换成 `UnfoldConv2d`（unfold + bmm），只影响训练。

## 阶段 0：环境（约 5 分钟）

1. 远端：`nvidia-smi --query-gpu=memory.used --format=csv,noheader; df -h /root/autodl-tmp | tail -1`。
   期望显存 < 100 MiB、可用 ≥ 15G。
2. 本地：`$env:DEEPSEEK_API_KEY.Length -gt 0`（若为 False，从 User 作用域注入：
   `$env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY","User")`）；
   `wsl -d Ubuntu -- docker ps --filter name=tp- --format "{{.Names}}"` 应无输出；
   代理 `http://127.0.0.1:7890` 可用（`curl.exe -x http://127.0.0.1:7890 -I https://pypi.org`）。
3. 阶段 0 失败：只能做不需要该资源的步骤（例如代理坏了仍可做阶段 1、2）。

## 阶段 1：RAFT 训练校准（远端约 1–2 小时）

材料：远端 `/root/autodl-tmp/RAFT-Benchmark-v1/`。参考 `controller-reference-r1/evaluation_entry.py`
（SHA256 必须是 `45610d8114e4f9311f83cf887d572ad5fa21cc69dd55c57c014cd97c837bf693`，不是就 FAIL
并 SKIP 整个 RAFT）。训练 oracle `oracle-bundle-train-three-r1-r6/`，推理 oracle
`oracle-bundle-inference-r1-r6/`。已有驱动 `run_verify_reference.py`、`run_verify_inference_v3.py`、
`run_verify_unfold_r1.sh`（照它的环境变量写法）。本地副本在 `D:\科研\RAFT-Benchmark-v1\controller\`。

照搬 SimCSE 的校准方法。先读：
`experiments\p2\simcse-bert-base-v1\train-v2-rule-calibration-20261029.md`、`loo_repeat_baseline.py`、
`make_pilot_task.py`、`make_small_negative_controls.py`、`run_negative_controls.py`；
FNO 的同类做法见 `experiments\p2\fno-v1\4090-pilot-20261103.md`。

1.1 参考复跑 12 次：用同一份绑定 TaskSpec（复用 oracle，不要每次新路径），trial 名
    `cal-r01` … `cal-r12`。每次只需目标端输出（`outputs.npz`），旧规则判 `PARITY_FAILED` 是预期。
    每次约 30 秒加编译；12 次全部能跑完（无 `RUN_FAILED`）才算 PASS。
1.2 建基线：用 `loo_repeat_baseline.py`（参数 `--k 8 --floor 1e-7`）对 12 次参考做留一检验并生成
    全量 judge-only 基线 `target-repeat-baseline.npz`，记录 SHA256。门槛：留一 0 误拒。
1.3 负对照 3 个：复制参考到新目录，各只改一处：
    `neg-gamma`（sequence loss `GAMMA` 0.8→0.85）、`neg-wd`（`WEIGHT_DECAY` 5e-5→5e-4）、
    `neg-gru`（`SepConvGRU._half` 里 `(1 - z) * h + z * q` 改成 `z * h + (1 - z) * q`）。
    用全量基线判定。门槛：3 个全部被拒。
1.4 生成 RAFT pilot 训练 TaskSpec：在 r6 训练 task 上加
    `train_envelope=train-repeat-envelope-v1-pilot`、`train_noise_baseline`、`train_noise_baseline_sha256`
    （照 `experiments\p2\fno-v1\prepare_train_trial.py` 的写法），再用参考跑一次隐藏 `verify`，
    门槛：`passed=true`。
1.5（不阻塞）Sintel 子集 EPE：`experiments\p2\raft-v1\eval_trained_states.py`，跑不通只记录。

1.1–1.4 全过 → 阶段 3 做 RAFT；否则 RAFT trial SKIP。推理部分已在 2026-11-03 通过 v3 verify
（543 张量 0 超限，max ratio 3.14），不需要重做。

## 阶段 2：FNO / LightGCN / SimCSE 推理任务（每个约 30–60 分钟）

不需要单独建 oracle：推理 TaskSpec 用 `noise-envelope-v3`，`verify` 会自动在远端生成 source 输出。
协议块与 RAFT 推理完全相同（见 `D:\科研\RAFT-Benchmark-v1\controller\run_verify_inference_v3.py`）：

```json
{"profile": "noise-envelope-v3", "repeats": 2, "timeout": 1800, "pass_k": 8.0, "floor_k": 8.0,
 "noise_floor": 0.0, "variants": [{"name": "fp64", "dtype": "float64"},
 {"name": "benchmark", "cudnn_benchmark": true}, {"name": "channels_last", "channels_last": true}]}
```

（`channels_last` 对非 4D 模型若报错，只去掉这一个变体，并在日志里写明。）

每个仓库：
2.1 素材上远端（新目录 `/root/autodl-tmp/<仓库>-Benchmark-v1/inference-r1/`）：source 源码、
    checkpoint、固定输入 `.npy`。
    - FNO：`D:\科研\FNO-Benchmark-v1\`（源码 `source/`，权重/输入见 `controller\train-inputs-4090-r2\`
      与 `controller\task-metrics-*`）；远端已有 `FNO-Benchmark-v1/source`。输入用 Darcy 16×16 的
      固定 4 个样本。
    - LightGCN：`D:\科研\LightGCN-Benchmark-v1\`（`source/`，checkpoint 见 `controller\` 与
      `experiments\p2\lightgcn-v1\train-ckpt-five-step-3080ti-20261102.md`）；输入为固定 16 个 user id。
    - SimCSE：`D:\科研\SimCSE-Benchmark-v1\`（`source/`、`assets/` 下的 unsup checkpoint 和 STS-B
      预分词输入）；远端 SimCSE 目录是空的，需要整体上传，先确认空间。
2.2 写推理 TaskSpec（照 `D:\科研\RAFT-Benchmark-v1\controller\inference-r1\task.json` 的结构）：
    target 入口 `evaluation_entry.py`、`load_inference_model`（SimCSE 用已有 `run.py:load_model` 约定，
    见其 `prompt.md`）。
2.3 正确参考：FNO 用 `D:\科研\FNO-Benchmark-v1\controller\target-reference-4090-r1`，LightGCN 用
    `experiments\p2\lightgcn-v1\train_target_reference.py` 对应的参考，SimCSE 用
    `D:\科研\SimCSE-Benchmark-v1\reference`。若参考只有训练入口，在**复制出的新目录**里补
    `load_inference_model()`（复用其模型，`eval` 模式），不改原目录。
2.4 参考跑隐藏 `verify` 两次。门槛：两次 `passed=true`。
2.5 负对照 1 个（复制参考，把输出乘 `1.001`）。门槛：被拒。

2.4、2.5 都过 → 阶段 3 做该仓库；否则 SKIP。

## 阶段 3：工作区与控制目录

照 `experiments\p2\fno-v1\prepare_train_trial.py` 写一个新脚本
`experiments\p2\workflow-4090-r1\scripts\prepare_<仓库>_trial.py`，生成 none/full 两组：

- 工作区 `D:\科研\<仓库>-Benchmark-v1\workspaces-wf-r1\{none,full}`：上游源码在 `upstream/`，
  空的 `evaluation_entry.py`（SimCSE 用其已有种子 `seed/` 的结构），`git init` 单提交，两组同一提交。
- 控制目录 `D:\科研\<仓库>-Benchmark-v1\controller\<仓库>-wf-{none,full}-r1`：`task.json`、
  `mcp-trial.json`、`mcp-token.txt`、`dsh.patch.yml`、`preflight.json`、`audit-rules.json`、
  `container-trial.json`，字段照 FNO 脚本。`feedback`：none=`diagnostic`，full=`localized`。
- 远端 harness 根用新目录 `/root/autodl-tmp/<仓库>-Benchmark-v1/harness-wf-r1`；
  `python_config` 显式写 python3.9-config；JITTOR_HOME 每组独立
  `jittor-cache-wf-r1-{none,full}`，并用参考预热一次。
- RAFT 的 `tasks` / `hidden_tasks` 同时包含 `inference` 和 `training` 两个任务。
- 预算与端口：

| 仓库 | 提示词 | 墙钟 | query | 端口 none/full |
| --- | --- | --- | --- | --- |
| RAFT 推理+训练 | `prompt-raft.md` | 14400 s | 15 | 18941 / 18942 |
| FNO 推理 | `prompt-fno-inference.md` | 7200 s | 10 | 18943 / 18944 |
| LightGCN 推理 | `prompt-lightgcn-inference.md` | 7200 s | 10 | 18945 / 18946 |
| SimCSE 推理 | `prompt-simcse-inference.md` | 7200 s | 10 | 18947 / 18948 |

提示词在本目录下，**不要修改**。两组用同一提示词。

门槛：两个工作区 `git status --porcelain` 无输出；工作区里搜不到 `task.json`、`oracle`、
`baseline`、`reference` 等隐藏文件。

## 阶段 4：trial（串行）

顺序：RAFT none → RAFT full → FNO none → FNO full → LightGCN none → LightGCN full →
SimCSE none → SimCSE full。被 SKIP 的仓库直接跳过。每个 trial：

1. 远端空闲（显存 < 100 MiB，无 `torchport` 进程），本地 docker 无 `tp-` 容器。
2. 启动：

```powershell
.\.venv\Scripts\python.exe experiments\harness\agents\run_container_trial.py `
  --controller "<控制目录>" --prompt experiments\p2\workflow-4090-r1\<提示词> `
  --proxy http://127.0.0.1:7890 --memory 5g
```

3. 结束后确认控制目录有 `container-runtime.json`、`audit.json`、`trial\result.json`、
   `trial\official-verify.json`，并检查远端 JITTOR_HOME 里只有 `cpython-39` 扩展。
4. 按 `experiments\p2\fno-v1\runbook-train-five-r1.md` 第 5 步的字段，把原始数字写进状态日志
   （只填数，不解释）。`agent.err.log` 里的 `double free`、`assert jit_utils.cc`、`fetch failed`、
   `TRANSPORT`、MCP 断连原样标注。

## 阶段 5：汇总

1. 拉回每个 trial 的远端证据（照 `D:\科研\FNO-Benchmark-v1\controller\pull_trial_evidence.py`）。
2. 在状态日志末尾写一张表：仓库、组、`passed`、`failure_code`、logical query 数、physical
   执行次数、agent 墙钟、模型步数、SKIP 原因。不写结论，结论由用户和 Claude 写。
3. 最后一行写 `WORKFLOW DONE`。
