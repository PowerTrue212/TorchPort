# Benchmark v1：多领域仓库迁移实验协议

状态：草案，2026-10-19；2026-10-26 修订 LLM 仓库（LoRA、nanoGPT 换为 Llama 3.2-1B、
Qwen2.5-0.5B）。仓库清单、方法和主模型已确认；B 的修订、训练 GATE 的具体参数和
各仓库 τ_j 仍待 pilot 后冻结。冻结前不跑正式 trial。

本协议在 [P2 评测协议](p2-evaluation-protocol.md) 的产物、成本和隐藏验收规则之上增加：
多仓库任务集、方法对照、训练检验和统一评分。两者冲突时以本文件为准，冲突点逐条列在
“与 P2 的差异”中。旧协议下的 CamoFormer 结果保持原判定，不改写。

## 研究问题

1. 同一仓库上，不同迁移方法（Harness 反馈有无定位信息、不同 Agent 系统、硬编码转换工具）的
   成功率和成本差异。
2. 同一方法在不同领域仓库上的表现差异，以及失败集中在哪些算子和语义上。

## 仓库

| 领域 | 仓库 | 人工 Jittor 迁移 | 特有内容 | 推理 | 训练 |
| --- | --- | --- | --- | --- | --- |
| LLM | meta-llama/llama-models（Llama 3.2-1B 参考实现） | 待核实（JittorLLMs 等） | RoPE scaling、GQA、RMSNorm、SwiGLU、KV cache；LoRA 微调（参数冻结、注入、checkpoint 分离） | 是 | LoRA 微调 |
| LLM | Qwen2.5-0.5B（`transformers` 的 `modeling_qwen2.py`，只取单模型文件及其最小依赖） | 待核实 | QKV bias、tied embedding、GQA；LoRA 微调同上 | 是 | LoRA 微调 |
| CV | CamoFormer | 无 | 分割、PVTv2 | 已完成 | 是 |
| CV | lucidrains/denoising-diffusion-pytorch | alexfanqi/denoising-diffusion-jittor | 噪声调度、采样循环、EMA | 是 | 是 |
| CV | AllenXiangX/SnowflakeNet（SPD） | AllenXiangX/SPD_jittor（作者官方） | 点云、Chamfer 距离 | 是 | 是 |
| Embodied | YanjieZe/3D-Diffusion-Policy（DP3） | Cuzyoung/Jittor-Embodied | 点云编码、UNet1D、动作扩散 | 是 | 是 |
| Embodied | tonyzhaozh/act | 无 | CVAE + Transformer、action chunking | 是 | 是 |
| Graph | LightGCN（独立 PyTorch 实现，不用 PyG） | 无 | 稀疏矩阵乘、embedding、负采样 | 是 | 是 |
| AI4S | neuraloperator/neuraloperator（FNO） | 无 | `fft`、复数张量、einsum | 是 | 是 |
| RL | vwxyzjn/cleanrl（PPO） | 无 | 分布、采样、log_prob、环境循环 | 是 | 是 |
| 多模态 | openai/CLIP（ViT-B/32） | uyzhang/JCLIP | 双塔、对比损失 | 是 | 微调 |

每个仓库在 TaskSpec 中冻结：源仓库 commit、入口、数据子集、checkpoint 来源、GPU 档位。
人工迁移只在本协议的 `verify` 通过后才作为对照；未通过时作为“人工迁移质量”数据单独报告。

LLM 两项（2026-10-26 修订）：

- 使用官方发布的预训练权重，不需要自行训练任务 checkpoint。LoRA 语义（冻结非 LoRA 参数、
  向注意力投影注入 LoRA、LoRA 权重与基座权重分开保存/加载）并入这两个仓库的微调任务，
  由 Harness 侧提供同一份最小 LoRA 实现规格，不再单独迁移 microsoft/LoRA。
- P_t 使用固定文本集上的困惑度和贪心解码 token 精确匹配率，评测脚本由控制端持有。
- 统一 fp32；V100 不支持 bf16，fp16 有溢出风险，不用于验收。
- GPU 档位：Qwen2.5-0.5B 推理与 LoRA 训练均可在 V100 上进行。Llama 3.2-1B 推理可在 V100 上
  进行；LoRA 训练 GATE 的 fp64 参考路径显存需求在 pilot 中实测，V100 放不下时整仓库改用
  3090/4090 档位（同一仓库所有方法同档）。
- 待核实：Llama 权重许可与下载方式；`llama-models` 参考实现是否依赖 fairscale 并行层；
  Qwen2 建模文件从 `transformers` 剥离后的最小依赖集合；两者是否存在公开 Jittor 人工迁移
  （JittorLLMs 等），存在则列入屏蔽名单。

未选及原因：OpenFold/OpenFold3/OpenVLA/DLRM-1TB/Graphormer-3D 需要 H100 或多卡；
Graphormer-slim 依赖 fairseq；PyG 版本依赖整个 `torch_geometric`；Audio、时序、推荐没有人工
Jittor 迁移；RetinaNet 被 SPD 替换。microsoft/LoRA 移出：官方没有 GPT-2 Small 的任务
checkpoint，E2E 评测链路需自建，P_t 无法定义；其已完成的 GPT-2 Small 前向/训练校准保留为
训练 GATE 的方法学证据。nanoGPT 移出：教学项目，代表性不足。MiniMind 未选：个人教学项目，
没有同行评审来源。DeepSeek-V3 未选：需要 H100，FP8 Triton kernel 超出本轮范围。

## 方法与实验矩阵

| 方法 | 说明 | 每格运行次数 |
| --- | --- | --- |
| DSH-none | DeepSeek V4.1 Flash + Harness，反馈 `diagnostic`（无 `trace`/`localize`） | 1 |
| DSH-full | DeepSeek V4.1 Flash + Harness，反馈 `localized` | 1 |
| Codex-none | Codex CLI，GPT 6 Sol + Harness，反馈 `diagnostic` | 1 |
| JTorch | 官方兼容层直接运行原代码 | 1 |
| 硬编码转换工具 | 官方 `pytorch_to_jittor` 等转换器 | 1 |

- 三种 Agent 方法都接 Harness，都能调用 `run`/`evaluate`/`verify`。不设“完全不接 Harness”的组。
- DSH-none 与 DSH-full 使用同一模型 ID、采样参数、基础 prompt 和预算，差异只来自定位反馈
  （`trace`、`localize`、首次分歧证据），用于回答定位反馈是否有效。
- Codex-none 代表另一种 Agent 系统（框架与模型一起变化），与 DSH-none 的反馈条件相同。两者的
  差异是“Agent 系统”之间的差异，不能归因于框架或模型中的单一因素。
- 不做重复：每个 (仓库, 方法) 只跑一次，共 11×3 = 33 次 Agent 迁移和 22 次基线运行。
- 每格 n=1，方法比较只做跨仓库的配对比较（仓库作区组，符号检验 / Wilcoxon），不对单个
  仓库下方法优劣的结论。
- JTorch 和转换器在某仓库无法 import 或运行时记 R_e=0，计入统计，不排除该仓库。
- 预算：每个 trial 的 wall time、Token、turn 和 oracle query 上限在 pilot 后冻结，所有方法相同。

## 环境

- 同一仓库的所有方法在同一种 GPU 上运行；GPU 型号、驱动、CUDA、双端框架版本写入 TaskSpec。
  换卡即 oracle 失效重建。不同 GPU 档位之间不比较 B，τ_j 和噪声标定按档位分别做。
- V100 主机（SXM2-16GB）：source 为 `/root/data-fs/APT/.venv-source`（Python 3.10，torch
  2.4.1+cu121），target 为 `/root/data-fs/APT/.venv-jittor`（Python 3.9，Jittor 1.3.11.0，CUDA
  11.3，sm_70）。与 AGENTS.md 中 CamoFormer V100 复现的版本（PyTorch 2.5.1 / Jittor 1.3.9.14）
  不同，CamoFormer 需在本环境重跑后纳入。
- V100 根分区已满，Jittor 缓存必须通过 `HOME` 或 `JITTOR_HOME` 指到 `/root/data-fs`。
  data-fs 约 15G 可用，大数据集需另行安排存储。
- 新租用的 RTX 4090（sm_89，驱动 535.216.03，CUDA 12.1）已连通；source 是 Python 3.11.10 /
  PyTorch 2.4.0+cu121，target 是独立 venv 中的 Jittor 1.3.11.0。该环境只作 pilot 校准，
  不与 V100 的 oracle 或结果混用；正式使用前仍需冻结依赖、负对照和 worker 隔离证明。
  3080/3090/H100 等其他档位的环境在使用前补充到本节。

## 验证代码隔离

none 组不能通过读代码学到“首次分歧定位”这类方法。为此所有 Agent 组（含 full）都只能通过
Harness 工具的返回值获得验证信息，看不到验证实现：

1. Agent 可见文件系统中没有 TorchPort/Jittor Harness 源码（`src/torchport/`，尤其 `diagnosis/`、
   `capture/tracing.py`、`verification/`、`comparison/`）、`experiments/harness/`、本仓库的
   AGENTS.md 和文档、oracle、hidden task、其他 trial 的工作区和证据。
2. Harness 工具由外部适配层（MCP 或 CLI 包装）在 Agent 进程之外调用，Agent 环境的 Python
   中不安装 `torchport` 包，`import torchport` 必须失败。
3. 远端 GPU 上 evaluator 的同步目录和 oracle 目录不对 Agent 的 Shell 开放；Agent 的远端运行
   使用独立账户或独立目录。
4. 公开网络屏蔽本项目自己的 GitHub 仓库（Jittor Harness / TorchPort，含 raw 地址），
   与人工迁移仓库一起列入屏蔽名单。
5. 每个 trial 从 candidate 工作区内启动新会话，不继承旧会话和记忆（BiRefNet none-r0 曾因
   cwd 为仓库根目录读到 AGENTS.md 和 `src/torchport`）。
6. 启动前试读上述路径和 `import torchport`，确认全部失败，结果写入 worker attestation。
   事后检查事件日志中的文件读取和 URL，命中者记 `scope_violation`，不计入有效 trial。

实现约定（2026-10-26）：

- 运行身份：每个 trial 在一个新的 Docker 容器中运行 Agent（`docker run --rm`），镜像由
  `experiments/harness/agents/container/Dockerfile` 构建，只含 node、DSH 和不带 torchport 的
  Python。容器只挂载本组 candidate 到中性路径 `/work`（可写）和本 trial 的 DSH patch（只读）；
  控制端仓库、控制目录、oracle、另一组工作区和宿主机凭据不挂载。两组使用同一路径 `/work`，
  路径不暴露组别。DSH 主目录在容器内新建，不继承宿主机会话与记忆。镜像摘要写入 attestation。
  远端 GPU 不向 Agent 开放 SSH，GPU 执行只经 Harness `run`/`verify`。参考宿主机为 WSL2
  中的 docker.io（`--network host`）；Linux 服务器用法相同。
- MCP：控制端账户用 `mcp_server.py --http 127.0.0.1:<port> --token-file <控制端文件>` 单独启动
  服务，容器以 host 网络经回环地址连接（DSH `streamable-http`）。每个 trial 一个端口和 token；
  容器内不运行也不可读 Harness 源码。stdio 方式只用于本机自测。
- 启动前探针：`preflight.py` 在同一镜像、同一挂载的容器内运行，试读宿主机路径、试
  `import torchport`、试连禁止 URL、确认 `/work` 可写，任一越界成功即不启动 trial；输出原样存入
  控制目录，作为 attestation 的依据。探针配置只临时挂载，不留在 `/work`。
- 金丝雀：`audit_trial.py plant` 在控制目录和 oracle 目录放置唯一标记文件，标记只记在控制端。
  不在另一组工作区放置，否则它会在另一组的 trial 中出现在 `/work` 里。容器隔离下这些目录
  本不可见；金丝雀用于证明这一点并发现挂载配置错误。
- 编排：`run_container_trial.py` 依次执行 plant → 启动 MCP → 容器内 preflight → 容器内 DSH →
  停止 MCP → scan → certify，全部原始输出写入控制目录。容器不继承宿主机代理；需要代理时用
  `--proxy` 显式传入，回环地址绕过代理，两组必须使用同一代理，代理地址写入
  `container-runtime.json`。URL 屏蔽暂不在容器出口拦截，
  靠事后审计与相似度检查。
- 事后审计：`audit_trial.py scan` 扫描 Agent 会话日志中的金丝雀、工作区外路径、禁止路径与
  禁止 URL，并记录模型步数。命中金丝雀或禁止项记 `scope_violation`；只出现工作区外普通路径
  （如解释器、系统目录）时列出，人工复核。

## 答案泄露

公开网络和检索保持可用。对有人工迁移的仓库（目前确认 4 个：diffusion、SPD、DP3、CLIP；
Llama/Qwen 待核实）：

1. 外部适配层屏蔽对应人工迁移仓库的 URL（含 GitHub、镜像和 raw 地址）。
2. 事后计算 candidate 与人工迁移的代码相似度；超过阈值的 trial 标记 `reference_leak`，单独报告，
   不删除。阈值在 pilot 中用无泄露 trial 的分布确定。

## 指标

最终输出 R_e、B_ℓ、P_t、E_m，并保存全部原始数据，公式变更时从原始数据重算，不重跑实验。

### R_e：是否跑通（0/1）

候选在冻结入口上完成推理（训练任务另需完成规定步数）记 1，否则记 0。

### P_t：任务性能一致性（0~100），决定最终验收 s

相同测试集、预处理和评测代码下复测 source 与 target：

```text
g_j = |m_j^target - m_j^source| / τ_j
P   = 100/J · Σ_j 1 / max(1, g_j)
```

τ_j 在正式实验前由 source 自身的重复/seed 波动确定（例如 3σ_source 与人工下限取较大值），
写入 TaskSpec。s=1 当且仅当所有 g_j ≤ 1。有可精确比较的离散输出（argmax、贪心解码 token、
阈值化 mask）时，额外报告精确匹配率。

### B_ℓ：逐层行为一致性（0~100），诊断用，不参与 s

当前公式（与 docs/temp/point1.png 一致）：

```text
d_ℓ = max_{样本,i} |y_ℓ,i^target - y_ℓ,i^source| / (a_ℓ + r_ℓ |y_ℓ,i^source|)
B   = 100/L · Σ_ℓ 1 / max(1, d_ℓ)
```

缺失、形状错误或非有限值的检查点记 0。检查点列表由 source 事先固定，挂在拥有参数的模块上。

待定修订（pilot 中两种同时计算，再决定）：

- 用隔离执行（喂 source 真实局部输入）计分，自然轨迹只做诊断，避免上游误差重复扣分。
- d_ℓ 改为噪声标定比值 `err(target, fp64) / (k · noise_ℓ)`，替代固定的 a_ℓ、r_ℓ。
- 优先使用真实 checkpoint，随机权重下条件数会放大正常数值差异。

B 只在同一仓库内跨方法比较；跨仓库比较时按仓库分别报告，不解释为模型能力差。

### E_m：迁移效率（0~100）

公式按 docs/temp/point2.png，暂不修改：

```text
C = 0.4·T/T0 + 0.4·N/N0 + 0.1·K/K0 + 0.1·F/F0
E = 100·s / (1 + C)
```

- T：代码迁移有效耗时，排除模型运行、编译及验证耗时。
- N：累计输入与输出 Token。
- K：Agent 工具调用次数，所有方法使用相同计数边界。
- F：统一检查下失败的候选代码版本数，同一版本重复失败只计一次。Agent 每轮结束时记录 candidate
  内容摘要；trial 结束后用隐藏 task 对每个不同版本各 verify 一次，失败数即 F。这样 F 不依赖
  Agent 主动调用 verify 的频率。事后 verify 的物理成本单独记账，不计入 Agent 成本。Agent 过程中
  开发 verify 失败的版本数另行保存。
- T0、N0、K0、F0 在正式实验前固定，不是运行上限。

JTorch 和转换器没有 Token 和工具调用，N=K=0，按实际值代入。

### 必须保存的原始数据

| 项 | 字段 |
| --- | --- |
| T | trial 墙钟总时间；按 迁移 / 模型运行 / 编译 / 验证 分段的区间事件；API 请求延迟 |
| N | 每次请求的 input、cached input、output、reasoning、provider total；模型 ID 与版本；请求与响应文本字节数 |
| K | 每次工具调用的事件（工具名、开始/结束时间、退出状态），Harness 工具单独标记 |
| F | 每个 candidate revision 的内容摘要和统一检查结果 |
| B | 每个检查点在每个样本上的 max_abs、p99、source 幅值、fp64 参考误差、噪声估计、形状和有限性 |
| P | 每个指标的 source/target 原始值、τ_j 及其标定数据 |
| 环境 | GPU、驱动、CUDA、双端 Python 和框架版本、TaskSpec 摘要 |

现有 events.jsonl 已覆盖 K、Logical/Physical cost 和 provider usage；T 的分段计时、请求文本
字节数和 B 的逐检查点原始量需要补齐，补齐前不跑正式 trial。

## 训练检验

训练过程是反复应用更新映射 `θ_{t+1} = U(θ_t, batch_t)`。沿训练轨迹逐步验证 U 等价，比直接
比较全量训练结果更严格：后者受轨迹混沌和 seed 波动影响，统计功效低，测不出小 bug。

1. 连续 N 步等价（训练 GATE，v1，`protocol.profile: train-step`）。两端从同一 oracle 初始状态
   出发，在同一冻结 batch 序列（输入第 0 维为步号）上连续训练 N 步，N 取 3~5。每步记录 loss
   和更新后的全部可训练参数，与 source 做噪声标定比较：fp64 与 alt 两条参考路径同样连续走 N 步，
   噪声基准包含同样的累积误差；缺少标定的 `output.*` 直接判失败。source 开启
   `torch.use_deterministic_algorithms(True)`，oracle 仍要求逐位稳定。v1 要求随机层关闭
   （dropout=0 等），不回放随机 mask。
   - 任务契约：两端入口提供 `optimizer(model)`（返回 optimizer，或 (optimizer, scheduler, ...)）和
     `step(model, state, *batch) -> loss`，一步内的 zero_grad、backward、裁剪、optimizer/scheduler
     step 由 `step` 完成。
   - 升级条件：若负对照在连续 N 步下漏检，再升级为教师强制（target 每步载入 source 的参数与
     optimizer 状态，只走一步，误差不累积；需要 target 提供 optimizer 状态加载接口）。
2. 精确检查（属于 GATE）。已实现：可训练参数集合、步数、每步学习率逐项相同。待实现：EMA、
   BN running stats、AMP scaler 状态；checkpoint 保存再加载后前向一致。
3. 统计等价（外部效度参考，不进 GATE）。只在训练快的仓库（FNO、CleanRL、LightGCN）上做：双端多 seed 训练到缩减预算，用 TOST 比较最终指标，等价区间取 τ_j。
   事先做功效分析，功效不足报告“未判定”。

RL 的训练等价使用固定 rollout buffer 上的 N 次 PPO update。

负对照（证明 GATE 能抓住训练 bug，pilot 中完成）：loss reduction 用 sum 代替 mean、AdamW 写成
Adam+L2、漏掉 `zero_grad`、梯度裁剪顺序错、调度器差一步、BN momentum 约定不一致。逐项报告
被哪一部分检出。

实现复用现有 capture/comparison/verification，不建第二套执行器：`capture/training.py` 执行训练步
和 source 噪声标定，oracle 冻结 `train_fp64.npz`/`train_alt.npz`，最终由 `judge` 统一判定。
`CAPTURE_VERSION` 已升到 4，旧 oracle 失效重建。以下 LoRA GPT-2 Small 记录是训练 GATE
的方法学校准，该仓库已移出正式清单。LoRA GPT-2 Small 的前向与四步训练
校准 candidate 已分别在 V100 上通过；负对照尚未完成，其他仓库尚未验证。
2026-10-24 V100 复测正确训练 candidate 四次均通过，`loss-sum`、
`schedule-shift`、`adam-l2` 三个负对照分别被数值 GATE、精确 `train.lr`、
数值 GATE 检出；
其余负对照及 checkpoint 回环仍未完成，见
[复测记录](../experiments/p2/lora-gpt2-small-v1/train/v100-calibration-20261024.md)。
RTX 4090 pilot 的前向通过，训练在同一冻结 oracle 下出现两次失败、两次通过，
见 [校准记录](../experiments/p2/lora-gpt2-small-v1/train/4090-calibration.md)；不能把复跑通过
覆盖原失败，也不能在当前门槛下视为稳定校准。

## 与 P2 的差异

- P2 允许 Agent 检索和引用公开同题实现且不使 trial 无效；本协议对有人工迁移的仓库屏蔽 URL，
  并对泄露 trial 加 `reference_leak` 标记。
- P2 比较 verdict/diagnostic/localized 三档反馈；本协议只用 `diagnostic`（none）和 `localized`
  （full）两档，`verdict` 不在本轮范围内。
- P2 按 replicate 统计 Success@Budget；本协议每格一次，只做跨仓库配对比较。

## 待定

- 各仓库 τ_j、T0/N0/K0/F0、预算上限。
- B 的两种计算方式取舍。
- GPT 6 Sol 与 DeepSeek V4.1 Flash 的 tokenizer 不同，E 中的 N 跨 Agent 系统比较时需同时报告
  文本字节数（原始字段已保存）。
- 相似度检测工具与阈值。
- SPD、DP3、ACT 的数据集体积与仿真环境依赖。
