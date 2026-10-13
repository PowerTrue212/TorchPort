# 2026-10-11 实验汇总

这份汇总沿用最初两张图的两层结构：先看运行成本，再看迁移结果。历史结果和这轮 v3 试跑分开，
避免把不同协议、不同候选和受污染的会话放在同一个成功率里。

## 先说结论

1. **PlantCamo 的配对试跑边界有效。** `none-r1` 实际是 diagnostic：DSH 能看到通用诊断，但看不到
   模块、张量名或首次分歧位置。它最终停在输出容器结构不一致；`full-r1` 使用 localized，E1 会话内
   通过，`max_ratio=1.2`。
2. **PlantCamo full-r1 的 E2 任务指标高度一致。** 250 张 PlantCAMO1250 测试图中，九项指标最大差
   `3.29e-8`，8-bit 预测最大只差 1，阈值 128 翻转比例 `2.61e-8`。这支持“候选行为正确”，但当前
   E2 的正式接受规则尚未冻结，所以这里称为任务级诊断证据，不改写成正式 E2 pass。
3. **VSCode-V2-T none-r1 的原始候选通过 E1，但实验条件失效。** DSH 后段四次显式设置
   `TP_FEEDBACK=localized`，并直接调用 runner 读取完整 verify 报告，其中还打印了
   `first_divergence`。因此这次不能计入 none 成功率，也不能与 full-r0 做公平成本比较。
4. **现在仍不能报告成功率。** PlantCamo 只有 diagnostic `0/1`、localized `1/1`；样本量不足。
   VSCode none-r1 又已受污染。下一步应先封住 shell 绕过，再按预注册顺序增加 replicate。

？
| 运行 | 条件 | 条件是否有效 | 活跃墙钟 | 步数 | Token（缓存读） | 费用 | TorchPort 调用 | 会话内 E1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| PlantCamo none-r1 | diagnostic | 是 | 77m45s | 222 | 3469 万（3439 万） | ¥1.48 | run 2 / evaluate 1 / verify 8 | fail：结构不一致 |
| PlantCamo full-r1 | localized | 是，重跑 | 31m58s | 95 | 927 万（906 万） | ¥0.70 | inspect 2 / run 3 / verify 1 | pass，ratio 1.2 |
| VSCode-V2-T full-r0 | localized | 历史 v3 基线 | 66m20s | 390 | 7949 万（7905 万） | ¥2.66 | inspect 1 / run 3 / evaluate 2 / verify 3 | official pass，ratio 1.074 |
| VSCode-V2-T none-r1 | diagnostic | **否：反馈泄漏** | 56m38s | 325 | 7433 万（7391 万） | ¥2.50 | 接口内 inspect 1 / run 7 / evaluate 2 / verify 8；另直接 verify 5 | 原始 pass；剔除 |

口径说明：

- 活跃墙钟取 DSH 的 `turn/start` 到 `turn/end`。VSCode none-r1 有两轮，表中相加后排除了中间
  14m44s 的人工等待。
- PlantCamo full-r1 表中只计磁盘清理后的正式重跑。此前还有一次 51 步、320 万 Token、¥0.28 的
  基础设施中止会话；它没有纳入主行，但没有从证据中删除。
- “TorchPort 调用”是原始调用数，不等于有效 oracle 查询。VSCode none-r1 的 5 次 direct runner
  verify 也属于查询，因此它的原始 Logical Query Cost 是 15（evaluate 2 + verify 13），并且其中
  4 次直接破坏了 feedback 条件。
- PlantCamo 两个 r1 尚未完成独立 official E1；表中是 DSH 会话内最后结果。VSCode full-r0 有正式
  E1。VSCode none-r1 即使今后独立复验通过，也无法修复已经发生的反馈泄漏。

## 新一轮迁移结果

| 运行 | Jittor 候选代码 | 参数 | E1 | E2 | 解释 |
| --- | ---: | ---: | --- | --- | --- |
| PlantCamo none-r1 | 725 行 | 1278/1278 | `STRUCTURE_MISMATCH` | 未运行 | 输出嵌套与冻结接口不同；这是结构失败，不能把 envelope ratio 当作通过分数 |
| PlantCamo full-r1 | 564 行 | 1278/1278 | pass，486 tensors，ratio 1.2 | 250 张；指标最大差 `3.29e-8`；像素最大差 1；翻转 `2.61e-8` | 当前最完整的 PlantCamo r1 结果 |
| VSCode-V2-T full-r0 | 1143 行 | 783/783 | official pass，561 tensors，ratio 1.074 | 6397 张；指标最大差 `2.75e-7`；像素最大差 1；翻转不超过 `6.01e-8` | 可作为 v3 历史基线，但不是 none-r1 的干净配对 |
| VSCode-V2-T none-r1 | 1143 行 | 783/783 | 原始 pass，662 tensors；最后允许的 diagnostic 工具输出 ratio 1.226 | 未运行 | localized 泄漏后得到的通过，不能算 none 成功 |

PlantCamo full-r1 的 E2 推理时间为 PyTorch `229.4s`、Jittor `466.8s`，这次约慢 `2.03x`。最初
图片里的 `1.3x` 属于旧的 full-r0 运行，不应用它替代本轮实测。

## 与最初两张图的关系

最初的四仓库汇总仍可作为历史概览：

| 仓库 | 历史运行 | 反馈 / 协议 | 墙钟 | 步数 | Token | 费用 | 最终任务级结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| CamoFormer | 10-05 试点 | 当时工具未接远端 GPU / v1 | 约 63m | 219 | 6060 万 | ¥2.58 | 实际正确；E2 最大指标差约 `1e-7` |
| BiRefNet-COD | none-r0 | diagnostic / v2 | 2h44m | 333 | 9150 万 | ¥6.15 | 实际正确；E2 最大指标差 `8.5e-8` |
| PlantCamo | full-r0 | localized / v2 | 2h40m | 234 | 5060 万 | ¥3.87 | 找到并修正 17 处 LayerNorm eps；修正后 E2 最大差 `5.6e-8` |
| VSCode-V2-T | full-r0 | localized / v3 | 66m20s | 390 | 7949 万 | ¥2.66 | E1 与 E2 均通过 |

这四行不能直接拿来估计反馈层级的成功率：协议不同，候选起点也不一致。新的 PlantCamo r1 才是同任务、
同 v3 协议、同初始候选的配对试跑；但每组仍只有一次。

## 数据文件与证据

- 机器可读运行表：[run-summary.csv](run-summary.csv)
- 机器可读结果表：[result-summary.csv](result-summary.csv)
- 绘图脚本：[plot.py](plot.py)
- PlantCamo full-r1 E2：[../plantcamo-full-r1/metrics.json](../plantcamo-full-r1/metrics.json)
- VSCode full-r0 E1：[../vscode-v2-full-r0/e1-verify-full.json](../vscode-v2-full-r0/e1-verify-full.json)
- VSCode full-r0 E2：[../vscode-v2-full-r0/e2-metrics.json](../vscode-v2-full-r0/e2-metrics.json)
- VSCode none-r1 会话结束后的完整报告：[../vscode-v2-none-r1/dsh-final-verify.json](../vscode-v2-none-r1/dsh-final-verify.json)
- PlantCamo r1 本地控制证据：`D:\tp-runs\_control\plantcamo-pair-r1\`
- VSCode none-r1 本地控制证据：`D:\tp-runs\_control\vscode-v2-none-r1\none\`
- DSH 会话原始记录：`C:\Users\31561\.dsh\sessions\`

最初两张图片中的历史数字由图片转录；VSCode full-r0 的精确 v3/E2 数字又用仓库 JSON 复核。空缺项
保留为空，不按 0 绘图或计算。
