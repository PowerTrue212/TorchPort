# SimCSE 训练 trial r7 交接（2026-10-29，第二版）

给接手的 Agent：none 组已完成，只需再跑一次 full 组（`full-r7-retry2`），汇报结果，
最后清理远端。不要再做规则诊断；规则已在下面冻结。

## 当前进度

| trial | 状态 | 是否计入 |
| --- | --- | --- |
| `simcse-train-none-r7`（diagnostic） | 完成。隐藏验收两次重复都通过，6,384 个包络判定张量零超限 | 计入（`pilot_only`） |
| `simcse-train-full-r7` 首次 | 远端磁盘写满，Harness 调用报 `No space left on device`，隐藏验收同步失败 | 不计入 |
| `simcse-train-full-r7-retry1` | Agent 运行中 MCP 连接断开，Logical Query Cost 为 0，`stop_reason=infrastructure_error`，Agent 退出码 1，容器清理报错 | 不计入 |
| `simcse-train-full-r7-retry2` | **待运行** | — |

retry1 的候选代码虽然通过了隐藏验收，但 Agent 全程没用上验证工具，不是 localized
条件下的结果，不能和 none-r7 比较。retry1 的 MCP 断开原因未查清；控制目录里的
`mcp.err.log` 为空。

结论记录在 [r7 结论](train-r7-pilot-outcome.md)（写于 retry1 之前，需要补上 retry1 和 retry2）。

## 已冻结的判定规则

- 名称 `train-repeat-envelope-v1-pilot`（修订 b）。TaskSpec：
  `D:\科研\SimCSE-Benchmark-v1\controller\train-v2-rule-calibration-20261029b\task-repeat-envelope-pilot-b.json`。
- 基数 `target-repeat-baseline.npz`，SHA256
  `0d54c6d44bb5eff13fc5481c1605e11be35ba2460072692ec3a205a7219a0bc1`，由 13 次控制端参考
  target 运行的两两最大差建立（脚本 `loo_repeat_baseline.py`，清单在同目录
  `target-repeat-baseline.json`）。
- 校准证据（同目录 `loo-check.json`）：13 次留一检验零误拒，最坏阈值使用率 0.516；
  7 个负对照全部检出，离边界最近的 `beta2-0p998` 超限 22 倍。
- oracle digest `ca1314b00d197ef04fffa7561b3c28545079847de22dbc6f6666dffe7bb82ab4` 与 r4、
  none-r7 相同。旧基数 `c22e70b5…` 已弃用，旧结果不改写。

## 远端现状（2026-10-29 晚）

- `/root/data-fs/SimCSE-Benchmark-v1/harness/` 下 16 个旧工作目录已删除。删除前逐个核对，
  全部文件都在 `controller/remote-archive-final/manifest.json` 里有哈希一致的本地副本
  （`verify.json` 0 个不一致）。
- 删除后 data-fs 剩约 28 GB（共享盘，别人也在占用，开跑前再查一次）。
  `SimCSE-Benchmark-v1` 剩 3.7 GB：`harness/weights-cache`、`assets/`、Jittor 缓存、`home/` 等，
  retry2 要用，不要删。
- retry2 会新建自己的远端工作目录并重建一次 oracle（约 10 分钟 GPU、3–4 GB 空间）。

## 无效或未使用的记录（保留，不删除，不计入结果）

- `controller/simcse-train-none-r5`、`simcse-train-full-r5`：旧基数，未启动。
- `controller/simcse-train-none-r6`：本机内存回收停掉编排，无审计、无隐藏验收。
- `controller/simcse-train-full-r6`：已准备，未启动。
- `controller/simcse-train-full-r7`、`simcse-train-full-r7-retry1`：见上表。

## 运行 retry2

前提：`DEEPSEEK_API_KEY` 已设置；WSL 中 Docker 可用；代理 `http://127.0.0.1:7890` 可用；
远端 `/root/data-fs` 剩余 ≥ 5 GB。若用 Claude Code 跑，必须用
`CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1` 启动。

先确认 retry1 的容器已经不在（`docker ps -a` 里没有 `tp-simcse-train-full-r7-retry1`；
若还在，`docker rm -f` 删掉）。

在 `D:\科研\TorchPort` 下用 PowerShell：

```powershell
$B = "D:\科研\SimCSE-Benchmark-v1"
$T = "$B\controller\train-v2-rule-calibration-20261029b\task-repeat-envelope-pilot-b.json"
$P = ".\.venv\Scripts\python.exe"

# 1. 干净工作区，seed 340b971
git clone "$B\seed-train\SimCSE" "$B\workspaces-train-r7-retry2\full"
git -C "$B\workspaces-train-r7-retry2\full" checkout 340b9713891b723ce8ec6f901049c618b5606b80

# 2. 控制目录（脚本核对 seed 提交、工作区干净、oracle digest）
& $P experiments\p2\simcse-bert-base-v1\prepare_train_trial.py --condition full `
  --trial-id simcse-train-full-r7-retry2 --port 18777 `
  --workspace-root "$B\workspaces-train-r7-retry2" `
  --task $T --remote-base /root/data-fs/SimCSE-Benchmark-v1

# 3. 运行（最长约 3 小时）；参数必须与 none-r7 相同
& $P experiments\harness\agents\run_container_trial.py `
  --controller "$B\controller\simcse-train-full-r7-retry2" `
  --prompt experiments\p2\simcse-bert-base-v1\prompt-train.md `
  --proxy http://127.0.0.1:7890 --memory 5g
```

若 `prepare_train_trial.py` 报工作区不干净或提交不符，先查原因，不要绕过检查。
若 Agent 运行中又出现 MCP 断开（`agent.jsonl` 里 `fetch failed`、工具调用失败），
记录为基础设施错误，先查 `mcp.err.log` 和编排进程是否被系统停掉，再决定是否 retry3，
并告诉用户。

## 结果怎么报

读 `container-runtime.json`、`audit.json`、`trial/result.json`、`trial/official-verify.json`，
按 [r4 结论](train-r4-pilot-outcome.md) 的表格汇报：预检/审计、Agent 退出码、步数、
Logical Query Cost（分工具计数、唯一 candidate 数）、Physical Evaluation Cost、Agent 墙钟、
隐藏验收结论和首个超限。

- 补写 `train-r7-pilot-outcome.md`：加入 retry1（无效原因）和 retry2 两列，再给出
  none-r7 与 full-r7-retry2 的对比。r4 记录不改。
- 两组都标 `pilot_only`（MCP 适配层只接受 `pilot_only`，隔离 worker attestation 尚未接上）。
- Jittor `double free`、`malloc(): memory corruption` 一类内存损坏单独标注，不当作 Agent
  实现错误。
- 首次超限位置不是已确认根因。

## 结束后清理远端（删除前必须问用户）

1. 用 `controller/cleanup-20261029/pull_evidence_incremental.py` 的做法拉回 retry2 新增的
   远端文件：已在 `remote-archive-final/manifest.json` 里且路径、大小未变的沿用旧哈希，
   新文件在远端算 SHA256 后下载并本地校验。脚本要求输出目录不存在，用新目录
   （如 `remote-archive-retry2`）；先读脚本确认参数。
2. 全部核对 0 不一致后，向用户申请删除整个 `/root/data-fs/SimCSE-Benchmark-v1`，确认后再删。
   `/root/data-fs/APT`（运行环境）不删。根分区 `/root/SimCSE-Benchmark-v1`（13 GB，r4 证据）
   不在范围内，需另行确认。

## 未提交的仓库改动

`prepare_train_trial.py`、`verify_train_reference.py` 的修改，新脚本 `loo_repeat_baseline.py`、
`make_pilot_task.py`、`make_small_negative_controls.py`、`run_negative_controls.py`、
`summarize_negative_controls.py`、`summarize_verify_runs.py`，以及 `train-r7-handoff.md`、
`train-r7-pilot-outcome.md` 均未提交。LightGCN 目录与 AGENTS.md 另有其他会话的改动，提交前分开确认。
