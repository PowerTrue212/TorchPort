# CLIP full-r3 pilot 结果

本次运行使用 `localized` 反馈。控制目录是
`D:\科研\CLIP-Benchmark-v1\controller\clip-full-r3`。原始 `agent.jsonl`、
`audit.json`、`trial/result.json`、`trial/official-verify.json` 和远端输出 NPZ 均保留。

## 技术结果

- 容器预检通过，Agent 退出码为 0，记录 66 个模型步骤。
- Harness 查询 6 次：`evaluate` 3 次、`trace` 2 次、`verify` 1 次。
- 冻结交付含 24 个文件。正式 `verify` 的推理任务通过，`strict_aligned=true`。
- 目标端加载 302/302 项权重，无缺失、多余或数值回读错误。
- 正式比较的最大绝对误差为 `3.62396240234375e-05`，不匹配比例为 0。
- 固定的 100 张 CIFAR-10 图像上，源端和目标端 top-1 均为 `89/100`。
  两端预测完全相同。`task-score.json` 保存逐图输出文件的摘要。

`task-score.json` 中 `P_t` 为 `null`，因为生成时没有传入已冻结的 `tau.json`
（`tau=0.01`）。两端 top-1 差为 0，按冻结 `tau` 计算 `P_t=100`。
本次未给出正式 `B_l` 或 `E_m` 分数。

## 审计与效力

运行进程在审计脚本修复前导入了旧代码。原 `audit.json` 把文本中的
`/mnt/data/...` 误匹配为禁止路径 `/mnt/d/`，于是把 `stop_reason` 记为
`scope_violation`。修复后的脚本对完整原始 `agent.jsonl` 重扫，
`audit-corrected.json` 显示 0 个金丝雀或禁止项命中、66 个模型步骤。
原 `audit.json` 和 `trial/result.json` 未覆盖。认证入口禁止重复认证，
因此原结果仍写 `valid=false`。此外该任务的 `claim_status=pilot_only`，
即使没有审计误报，也不能把这次运行写成正式有效 trial。

结论：**技术上的固定推理验收通过；正式 Benchmark v1 效力未判定。**
