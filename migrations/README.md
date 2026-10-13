# 真实仓库迁移工作区

这里集中保存真实 PyTorch 仓库迁移所需的仓库专用材料。TorchPort 核心不依赖这些目录；
它们是实验协议、提示词、适配器、本地工作区和仓库专用证据的容器。

| 目录 | 内容 |
| --- | --- |
| `CLIP/` | OpenAI CLIP 的冻结迁移协议与历史三方法实验 |
| `RAFT/` | RAFT 的冻结迁移协议、历史结果导出入口和本地工作区 |
| `CamoFormer/` | CamoFormer 受控迁移、校准和任务级指标材料 |
| `apt/` | Additive Prompt Tuning 的完整训练迁移协议 |

每个目录只提交小型协议代码和可审计材料。下载的源码、数据集、checkpoint、candidate
工作区、运行缓存和大型原始输出由根目录 `.gitignore` 排除。跨仓库的评测控制器、结果汇总
和图表仍放在 `experiments/`；TorchPort 的通用实现仍放在 `src/torchport/`。
