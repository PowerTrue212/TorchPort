# 隐藏最终评测

两个条件（`none` / `full`）在运行结束后由同一评测器判定，评测器与 oracle 不进入任何 Agent 工作区。
正式验收必须在 Agent 会话停止后、由 Harness controller 或实验者在独立终端运行。完整 JSON 只能写入
评测方结果目录，不能返回给 Agent，也不能放入 Agent 工作区。步数、Token 和费用从 Agent 会话原始记录
读取，不从验收输出推断。

## E1：TorchPort verify

先冻结 Agent 的完整仓库快照、文件 manifest 和 Git diff，再用本次物化的
`_tasks/<repo>-<condition>-r<k>.json` 对快照运行 `verify`。`run.py` 只是统一入口。
当前新运行冻结为 `noise-envelope-v3`、source 关闭 TF32、fp32、5 个 source 变体、
candidate 重复 3 次、`pass_k=2`、`floor_k=4`、`noise_floor=auto`。`verify` 是唯一最终判定入口。

开发期 Agent 调用的 `verify` 计入 Logical Query Cost。会话结束后的这一次独立 verify
计入评测方 Physical Evaluation Cost，不计入 Agent query。`none` 的开发期响应只有 diagnostic 字段；
评测方仍保存完整原始报告，但绝不能把 `first_divergence`、`first_marginal`、逐张量列表、模块构造参数差异
或自由文本问题描述返回给 Agent。

## E2：任务指标

在各仓库测试集上用官方指标计算 Jittor 候选的结果，并与 `heldout/<repo>/smoke/metrics-torch.json`（PyTorch 参考）比较。

- BiRefNet：CHAMELEON、NC4K、CAMO、COD10K，参考为 fp16 autocast；候选以 fp32 运行，差异须在报告中注明。
- VSCode-v2：COD10K、CAMO、NC4K。
- PlantCamo：PlantCAMO1250 test。

## E2 噪声包络（待做）

固定容差不能区分后端数值差异与实现错误，E2 需要 source 自身的变体指标运行。每个仓库至少需要：

1. source fp32，TF32 关闭，重复 2 次（重复性）。
2. source fp64（或 cpu）全测试集指标。
3. source 开启 cudnn benchmark 的指标。
4. source channels_last 的指标。
5. BiRefNet 额外：source fp16 autocast（官方协议）与 fp32 各一次。

据此得到各指标的 source 变动范围，候选落在范围（含倍数 k）内才算 E2 通过。k 在跑 agent 之前冻结。
