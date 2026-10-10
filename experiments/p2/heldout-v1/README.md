# heldout-v1 实验材料包

三个 held-out COD 仓库（BiRefNet-COD、VSCode-v2-T、PlantCamo）× 两个条件（`none` / `full`）的启动材料。
唯一自变量：agent 是否拥有六个 TorchPort 工具。任务正文、预算、停止规则完全相同。

2026-10-09：从 `plantcamo-full-r0` 起使用 prompt v2（`k=2` 提交下限与三级访问规则），
工作区移到仓库外；`birefnet-none-r0` 保留旧提示词，作为历史记录。

## 目录

```text
prompts/    body.md + tools-none.md / tools-full.md -> build.py -> none.md / full.md
tasks/      <repo>.json：noise-envelope TaskSpec 草稿（仅评测方使用）
inputs/     make_inputs.py 与生成的固定单图 .npy
workspaces/ materialize.py 与 `_tasks/`；agent 工作区可用 `--out` 放到仓库之外
evaluator/  隐藏最终评测说明（E1 / E2）
heldout/<repo>/adapter/torchport_adapter.py  PyTorch 侧 adapter（在 source/ 之外）
```

## Agent 远端目录

Agent 只使用 `scripts/remote/agent_ssh.py` 的 `run` 和 `put`。启动 `dsh web` 前，实验者应在
同一个 cmd.exe 窗口设置本次运行的唯一远端目录：

```bat
set TP_AGENT_REMOTE_DIR=/root/autodl-tmp/agent/plantcamo-full-r0
set TP_AGENT_AUDIT=D:\tp-runs\plantcamo-full-r0\.torchport\agent_ssh.jsonl
```

随后用与本次 TorchPort 调用完全相同的 task 路径准备 `staged/`。该 task 必须先运行过
`torchport_inspect` 或一次 `verify`，让远端已有冻结输入和权重缓存：

```bat
E:\Anaconda\python.exe scripts\remote\stage_agent_dir.py plantcamo-full-r0 experiments\p2\heldout-v1\workspaces\_tasks\plantcamo-full-r0.json
```

脚本把冻结输入和官方权重硬链接到远端工作目录的 `staged/`，并写入带 shape 和 sha256 的
`README.txt`。`agent_ssh.py` 的检查只是减少直接读取敏感路径的机会，不是 root 用户的安全
沙箱；每次运行后仍须检查 `TP_AGENT_AUDIT` 指向的 JSONL 日志。

## 启动步骤

1. 重新渲染提示词（会校验两份只在工具段不同）：
   `E:\Anaconda\python.exe experiments\p2\heldout-v1\prompts\build.py`
2. 生成工作区，例如：
   `python experiments\p2\heldout-v1\workspaces\materialize.py --out D:\tp-runs --repos plantcamo --conds full --k 0`
   默认不限 token / 200 轮 / 180 分钟。每个工作区内有 `PROMPT.md`（占位符已替换），任务副本仍写入
   `workspaces/_tasks/`。
3. 用户自己运行 `dsh web`，把生成的 `<repo>-<cond>-r0` 目录选为工作区，把该目录的 `PROMPT.md` 内容作为首条消息。
4. `none`：DSH 不安装 TorchPort 插件（或用不含该插件的 profile）。`full`：安装 `integrations/dsh` 插件，
   其 `task` 参数指向 `workspaces/_tasks/` 下对应任务文件。
5. 每次运行结束后保存 DSH 事件日志与 Token 用量，再按 `evaluator/README.md` 统一评测。

## 启动前必须核对

- [ ] `tasks/*.json` 中 `_unverified` 字段已逐项处理（尤其 target 端 repo 路径、远端路径、权重路径）。
- [ ] `full` 条件的 task 路径、`root`、`trial` 与工作区对应；远端 GPU 可通过 `~/.tp_ssh.env` 连接。
- [ ] `none` 条件的 DSH 确实没有加载 TorchPort 插件，且 agent 无法读到仓库根目录其他位置（工作区之外）。
- [ ] 工作区内没有 Jittor 参考实现、oracle、隐藏输入、evaluator（materialize 之后再 grep 一遍 `jittor` 以外的泄露）。
- [ ] 权重只以路径引用形式给出（`WEIGHTS.md`），agent 工作区不含 .pth。
- [ ] PlantCamo：`oracle-src/torchport_adapter.py` 只在构造时跳过硬编码的 backbone 权重读取；需确认完整 checkpoint 严格加载且无缺失键。
- [ ] VSCode-v2：非严格加载（64 个其他任务分支的 unexpected key，需剥 `module.` 前缀）与 TaskSpec 的权重匹配规则一致。
- [ ] BiRefNet：官方为 fp16 autocast，TaskSpec 用 fp32；E2 指标需注明协议差异。
- [ ] 输入 .npy 的预处理：BiRefNet 的 cv2.resize 本地用 numpy 复刻（本机无 cv2），需在有 cv2 的环境对比一次。
- [ ] 三个仓库的 source-only smoke 已在目标 GPU 上通过，Jittor 端环境（jittor 版本、CUDA）已确认。
- [ ] 各仓库每个条件的预算 `{{BUDGET_TOKENS}}`/`{{MAX_TURNS}}`/`{{WALL_MINUTES}}` 一致。

## 协议 v3 冻结（2026-10-10）

从本日起所有新 held-out 运行使用 `tasks/*.json` 中的 `noise-envelope-v3`：repeats 3，`pass_k=2`，
`floor_k=3`，`noise_floor=auto`（target 环境按 source 最大全局归约形状实测 N_env），变体
benchmark / fp64 / cpu / channels_last / batch2(distinct)。停止规则见 `prompts/body.md`：
`e1_status` 为 `pass` 或 `marginal` 才可提交。之前的运行（birefnet-none-r0、plantcamo-full-r0）
在 v2 下进行，只能用 v3 事后重验，不能与 v3 运行混在同一效率统计中。

冻结依据（校准集，远端 `tp-work/v3all`，报告副本见 `results/`）：

| 候选 | 预期 | v3 结果 | max_ratio |
| --- | --- | --- | --- |
| PlantCamo Agent 原候选（17 处 eps 写反） | fail | fail，首次 `backbone.norm1` | 10.9 |
| PlantCamo eps 修正版 | pass | pass | 1.64 |
| CamoFormer 正例 | pass | pass | 0.66 |
| CamoFormer gelu_tanh / ln_eps / mask_residual | fail | fail，首次均为注入点 | 107 / 62 / 4.7e5 |
| BiRefNet none-r0（E2 326 张指标差 ~1e-8） | pass/marginal | marginal | 2.86 |

已知边界：k 只用这 7 个候选估计；3 到 60 之间没有负对照；覆盖率 67%~94%；单张输入；
函数式全局归约（`x.mean((2,3))`）不进入 N_env 形状推断。
