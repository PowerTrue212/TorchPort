# TorchPort DSH 插件

该包把 TorchPort 的确定性运行、数值验收和结构化失败结果注册为 DeepSeek Harness
工具。DSH 负责阅读 PyTorch 仓库、编写 Jittor candidate 和选择修复；TorchPort
只执行冻结任务并返回可审计证据。

插件不包含自动代码转换器、仓库入口猜测或模型专用适配器。

该插件直接映射 TorchPort 的六个公开命令。正式 P2 的三组仍然
必须经过 `experiments/harness/`，由 Gateway 统一限制文件权限、计算 query/physical
cost 并隔离最终认证。不能直接运行本插件后依靠 prompt 要求模型忽略多余字段。

## 开发与测试

要求 Node.js 22、可调用的 DSH，以及已安装或可通过 `PYTHONPATH` 导入的 TorchPort。

```bash
cd integrations/dsh
npm install
npm test
npm run pack:check
```

## 安装到 DSH Web

```bash
cd integrations/dsh
npm install
npm test
dsh plugin --profile web add "$PWD"
```

安装一次并在 DSH Web 中把 TorchPort 根目录选为工作区后，以后只需运行 `dsh web`。
插件默认使用工作区中的 `.venv` 和 `src`，并通过
`scripts/remote/dsh_torchport.py` 把 TorchPort 命令发送到 `~/.tp_ssh.env` 指定的 GPU。
无需在启动 `dsh web` 前手工加载该文件。需要改为本机执行时，把
`TORCHPORT_RUNNER` 设为空；也可用 `TORCHPORT_PYTHON`、`TORCHPORT_PYTHONPATH` 和
`TORCHPORT_TRACE_DIR` 覆盖其他路径。

包内 `torchport.patch.yml` 会自动插入 `torchport` 插件行。API 密钥只通过进程环境或
DSH 凭据系统提供。

## 工具

| 工具 | 职责 |
| --- | --- |
| `torchport_inspect` | 查看公开任务契约 |
| `torchport_run` | 运行 source 或 target，生成运行记录 |
| `torchport_evaluate` | 读取既有产物并判定，不启动模型 |
| `torchport_trace` | 采集双端中间结果并比较 |
| `torchport_localize` | 从 trace 报告提取首次分歧 |
| `torchport_verify` | 冻结 oracle 并重复验收 target |

外部命令以参数数组启动，不经过 shell。每次 Python 工具调用都会追加 JSONL 轨迹；
插件只转录内核的 `passed`、`failure_code` 和证据，不维护第二套验收规则。

## 闭环

```text
DSH 阅读仓库并编写 candidate
              ↓
       torchport_verify
              ↓
     通过 ─────────→ 验收报告
              │失败
              ↓
 torchport_trace / torchport_localize
              ↓
        DSH 修改后重试
```
