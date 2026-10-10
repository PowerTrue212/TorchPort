你可以使用 DSH 自带的文件浏览、代码搜索、编辑、Shell 和 Python 能力，在工作区内读、改、运行代码，自行写测试和最小复现。

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
