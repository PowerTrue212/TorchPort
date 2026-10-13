# 工作区物化

`materialize.py --k <k>` 为每个 repo × condition 创建 `<repo>-<cond>-r<k>/`：

- `heldout/<repo>/source/` 的完整副本（PyTorch 官方代码）。
- `torchport_adapter.py`（Jittor 侧稳定入口，包装 `submission.load_model`）和空的 `submission.py` 骨架。
- `reference_torch_adapter.py`：PyTorch 侧 adapter 的只读参考。
- `PROMPT.md`：由 `prompts/none.md` 或 `full.md` 渲染（预算占位符已替换）。
- `WEIGHTS.md`：权重路径引用，不复制权重。

不包含：Jittor 参考实现、oracle、隐藏输入、evaluator、checkpoint。
任务文件写到 `_tasks/`，位于所有工作区之外。已存在的工作区不会被覆盖。

当前已生成 k=0：birefnet / vscode-v2 / plantcamo × none / full，共 6 个。
