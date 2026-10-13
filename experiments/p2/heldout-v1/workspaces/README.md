# 工作区物化

`materialize.py --k <k>` 为每个 repo × condition 创建 `<repo>-<cond>-r<k>/`：

- `heldout/<repo>/source/` 的完整副本（PyTorch 官方代码）。
- `torchport_adapter.py`（Jittor 侧稳定入口，包装 `run.load_model`）和薄的 `run.py` 骨架。
- `reference_torch_adapter.py`：PyTorch 侧 adapter 的只读参考。
- `PROMPT.md`：由 `prompts/none.md` 或 `full.md` 渲染（预算占位符已替换）。
- `WEIGHTS.md`：权重路径引用，不复制权重。

工作区是一个独立 Git 仓库。Agent 可以迁移仓库中的推理代码；`run.py` 只作为统一入口。
不包含：Jittor 参考实现、oracle、隐藏输入、evaluator、checkpoint。
任务文件写到 `_tasks/`，位于所有工作区之外。已存在的工作区不会被覆盖。

物化器不会覆盖已有工作区。旧版已经生成的工作区可能仍是单文件入口，不能直接当作新协议 seed；
新实验应使用新的 `--k` 或新的 `--out`，并确认根目录包含 `run.py` 和初始 Git commit。
