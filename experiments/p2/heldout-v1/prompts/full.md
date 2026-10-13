# {{REPO_NAME}}：从 PyTorch 迁移到原生 Jittor

你在一个独立的 Git 工作区中工作。当前目录是 {{REPO_NAME}} 官方 PyTorch 仓库的完整副本，
并附带一个空的 Jittor 入口。任务是迁移仓库中的推理实现，而不是只写一个提交文件。

## 开始前

先自行判断当前条件能否完成迁移。你可以阅读仓库和公开资料，并检查 Python、依赖、GPU 或 NPU、
显存、内存、磁盘、数据和 checkpoint。你也可以编写探测脚本或普通测试。发现问题后，自行决定修复、
下载公开依赖、使用已有资源、向用户说明阻塞，或者在证据足够时继续。Harness 只记录过程，不替你作判断。

## 硬性要求

1. 目标执行路径必须是原生 Jittor，不得通过 jtorch 或 PyTorch 兼容层代跑，不得硬编码输入输出，
   不得放宽误差标准。未进入目标推理路径的训练或参考文件可以保留。
2. Jittor 模块的参数名必须与 PyTorch `state_dict` 的键完全一致（含嵌套前缀、buffer 名）。权重由评测方加载，
   你的实现必须无需任何修改即可接收官方 state_dict。参数形状、数量也必须一致，多参数、少参数都算失败。
3. 迁移仓库并实现候选入口：
   - 把推理实现放在正常的仓库模块中，按需要修改、增加、删除或移动仓库文件。
   - 在 `run.py` 中实现 `load_model() -> jt.Module`。`run.py` 只负责导入和构造迁移后的仓库实现，
     不能容纳完整模型实现。
   - 返回的模块 `__call__` 接收一个 NCHW float32 图像 `{{INPUT_SHAPE}}`，返回元组 `tuple[jt.Var, ...]`，
     其结构、顺序、形状必须与 PyTorch 侧 `torchport_adapter.py` 的返回一致。
   - 根目录的 `torchport_adapter.py` 已是稳定入口，不要修改它。
   - {{ADAPTER_NOTE}}
4. 不要修改根目录的 `torchport_adapter.py`，不要读取 hidden task、oracle 或最终 evaluator。
5. 本次只要求迁移推理路径（eval、batch 推理、fp32）。训练和损失函数不要求迁移；数据加载与评测脚本
   是否修改由仓库正常推理是否需要它们决定。
6. 最终交付物是完整仓库。Harness 会保存整仓库快照、文件 manifest 和相对初始 Git 提交的 diff，
   并在快照上执行隐藏验收。

## 访问边界

访问规则分为三级：

1. 以下行为会使本次运行无效：绕过已提供工具，直接导入或执行 TorchPort 库；直接读取 oracle 或 evidence 内容、
   远端 verify 报告 JSON 或 envelope 数据；读取其他工作区或 evaluator。
2. 以下行为禁止，并会记录为实验偏差：读取 TorchPort 源码（`src/torchport/`，以及除调用 `ssh_exec.py` 之外的
   `scripts/remote/`）、仓库文档或 `AGENTS.md`。
3. 允许访问：当前工作区、代码 Agent 自带的代码与网络工具、已提供的 TorchPort 工具、你自己的远端工作目录，
   以及你建立的 PyTorch 与 Jittor 对照测试。正式实验若冻结了网络或挂载边界，以启动说明为准。

## 预算与停止规则

- Token 预算：{{BUDGET_TOKENS}}
- 最大对话轮数：{{MAX_TURNS}}
- 最长墙钟时间：{{WALL_MINUTES}} 分钟
- 以下任一情况出现即停止：预算耗尽、轮数用尽、时间用尽、你确认存在无法在当前环境内解决的阻塞，或你认为已完成。
- 不要在结束前反复请求用户确认，自主工作直到停止。
- `verify` 报告中的 `e1_status` 是数值验收的三档结论：`pass`（正式通过）、`marginal`（未正式通过，
  但已达到可提交底线，最终以任务指标为准）、`fail`（任务还没有完成）。只有 `e1_status` 为 `pass` 或
  `marginal` 时才可以结束并提交；为 `fail` 时继续工作，直到预算、轮数或时间用尽。
- `max_ratio` 是所有张量、所有重复运行中 `rel(T,R) / max(D, N_env)` 的最大值：`D` 是 PyTorch 合法运行之间
  的差异，`N_env` 是当前 Jittor 环境自身的运行间波动下限。`max_ratio <= 2` 为 `pass`，`<= 4` 为 `marginal`。

## 工具

<!-- TOOLS:BEGIN -->
你可以使用代码 Agent 自带的文件浏览、代码搜索、编辑、Shell、Python、Git 和网络检索能力，
在工作区内读、改、运行代码，自行检查环境并编写测试和最小复现。

远端 GPU（RTX 3080 Ti，已装 PyTorch 和 Jittor）可以直接使用：

- 运行命令：`D:\科研\TorchPort\.venv\Scripts\python.exe D:\科研\TorchPort\scripts\remote\agent_ssh.py run "<命令>"`
- 上传文件：`D:\科研\TorchPort\.venv\Scripts\python.exe D:\科研\TorchPort\scripts\remote\agent_ssh.py put <本地文件> <远端路径>`
- 你的远端工作目录：`/root/autodl-tmp/agent/{{WORKSPACE}}/`，这是唯一允许访问的远端目录。
- `staged/` 中已有冻结输入和官方权重，你可以用它们自行比较 PyTorch 与 Jittor。
- PyTorch：`/root/miniconda3/bin/python`
- Jittor：运行前设置环境变量
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python_config_path=/root/miniconda3/bin/python3.12-config use_cutlass=0 conv_opt=1 JITTOR_HOME=/root/autodl-tmp/agent/{{WORKSPACE}}/.jittor`，
  再用 `/root/autodl-tmp/jt-env/bin/python`。
- 不要访问远端 `/root/autodl-tmp/agent/{{WORKSPACE}}/` 以外的任何目录，不要读取任何凭据文件。

此外可以使用六个 TorchPort 工具（任务路径 `task`、记录目录 `root`、实验编号 `trial` 见启动说明）：

- `torchport_inspect`：查看公开任务要求；
- `torchport_run`：运行 PyTorch 源实现或当前 Jittor 候选实现；
- `torchport_evaluate`：比较已经生成的运行产物；
- `torchport_trace`：采集并比较两端的中间结果；
- `torchport_localize`：从 trace 报告提取首次分歧；
- `torchport_verify`：按冻结任务做数值验收。

验收结果包含逐层误差和首个超出允许范围的位置。
工具返回的原始结果就是证据，请据此决定下一步。`torchport_verify` 是普通验收的判定入口，但最终评测仍由评测方在你结束后完成。
<!-- TOOLS:END -->

## 结束时的汇报

用中文说明：修改了哪些文件、关键设计依据、你认为的当前状态、仍未解决的问题。不要声称“已通过”，最终判定由评测方完成。
