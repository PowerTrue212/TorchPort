# 验收任务

`tasks/` 保存公开、可复现的 TorchPort TaskSpec、源端 fixture 和目标端 candidate。
任务必须显式声明两端入口、模型、输入、设备、解释器和验收协议；TorchPort 不猜测
未知仓库的训练入口，也不自动改写模型代码。

模型迁移由 Agent 完成。正式 task 只负责回答：在冻结输入、权重、协议和运行环境
下，Agent 提交的 candidate 是否真实运行、完整加载状态并通过数值验收。

下载的 checkpoint 和大型运行产物放在 `artifacts/` 或远端数据盘，不提交到 Git。
