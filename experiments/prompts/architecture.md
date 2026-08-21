# TorchPort Figure Refinement Prompt

请在当前这张 TorchPort 方法图的基础上继续细化，不要重新设计结构，不要换成新的架构图。
保留现有布局：左侧输入，右侧黑框包围的主方法区域，主流程为
`DSH / Agent -> First divergence -> Repair -> Aligned Jittor`，底部是 dashed `Iterate` 回路。

## Overall Background

- 整体背景改为浅灰色，不是黑色、不是深灰色。
- 推荐背景色：`#F2F2F2`、`#F4F4F4` 或非常淡的 paper gray。
- 背景可以有极轻微纸张质感或中心亮度变化，但必须整体接近浅灰白。
- 不要纯黑背景，不要暗色舞台感，不要蓝色科技背景。

## Right Main Frame

- 右侧主方法区域必须用一个清晰的黑色矩形边框包起来。
- 黑框内部为白色或接近白色。
- 黑框不要圆角，不要阴影，不要渐变。
- 黑框线宽约 2.5 到 3.5 px，和图中箭头线条协调。
- 黑框要比当前更宽，向右拉长，给 `Aligned Jittor` 和右侧虚线回路更多空间。
- 黑框整体占画面宽度约 75% 到 78%，高度约 70% 到 75%。
- 黑框左边缘距离左侧输入胶囊保持清楚间隔，不要贴太近。
- 黑框右边缘距离画布右侧保留少量边距，不要显得拥挤。

## Keep Existing Architecture

不要改变当前的基本架构，只做比例和视觉细化：

1. 左侧输入区：
   - 保留小人图标。
   - 保留两个输入胶囊：
     `PyTorch repo`
     `weights + config`
   - 输入胶囊为淡蓝色，黑色描边，圆角胶囊。
   - 文字使用大号 serif 字体。
   - 输入到黑框内 `DSH / Agent` 使用粗黑箭头。

2. 黑框内主流程：
   - 保留四个模块：
     `DSH / Agent`
     `First divergence localizes mismatch`
     `Repair`
     `Aligned Jittor y_{t+1}`
   - 这四个模块横向排列，垂直居中略偏上。
   - 用粗黑实线箭头连接。
   - 箭头要短而有力，不要过细。

3. 底部回路：
   - 保留从 `Aligned Jittor` 返回 `DSH / Agent` 的黑色虚线回路。
   - 虚线回路沿黑框内部底部走线。
   - `Iterate` 放在虚线中间，字号大，serif。
   - 虚线不要贴到黑框边界，底部留出足够白边。

## Module Refinement

`DSH / Agent` 模块：

- 保持淡紫色竖向矩形。
- 黑色描边。
- 文字写：
  `DSH /`
  `Agent`
- 内部只放简洁图形，例如放大镜和 trace 节点。
- 不要放小段解释文字。

`First divergence` 模块：

- 保持最大的粉色圆角矩形。
- 位置在黑框中央偏左，视觉中心明确。
- 文字只写两行：
  `First divergence`
  `localizes mismatch`
- 字号比其他模块更大。
- 不要写 `semantic`，不要写长句。

`Repair` 模块：

- 保持淡紫色竖向矩形。
- 文字只写：
  `Repair`
- 内部可放简洁扳手图标。
- 不要写 `Repair Model`，避免文字太多。

`Aligned Jittor` 模块：

- 保持桃色圆角矩形。
- 由于右侧黑框要拉长，这个模块可以略向右移动。
- 文字写：
  `Aligned Jittor`
  `y_{t+1}`
- 字号大，居中。

## Bottom Labels

保留底部四组标签，但更简洁、更大：

- DSH 下方：`Evidence` / `p_ev`
- 粉色框下方：`Divergence` / `fd(y_t)`
- Repair 下方：`Repair` / `p_r`
- Output 下方：`Output` / `y_{t+1}`

要求：

- 底部标签使用大号 serif。
- 标签与上方模块垂直对齐。
- 标签不要太靠近虚线回路。
- 标签之间留白均匀。

## Typography

- 使用 Times New Roman、Computer Modern 或 Georgia 风格的 serif 字体。
- 所有文字都要比当前版本略大。
- 不要使用 sans-serif。
- 不要使用过多小字。
- 数学符号使用 italic math style。

## Color Palette

严格使用当前图接近的柔和配色：

- Outer background: light gray `#F2F2F2`
- Main frame fill: white `#FFFFFF`
- Input capsules: pale blue `#EAF4FF`
- Agent / Repair: pale lavender `#DDD1EF`
- Divergence: soft pink `#E9AFC2`
- Output: pale peach `#F5D6C4`
- Lines / borders / text: black `#000000`

## Negative Prompt

不要重新设计图。不要改变为三栏论文图。不要改成纯黑背景。不要取消右侧黑框。不要把右侧
主区域做成无边框白画布。不要增加模块数量。不要加入 scan / plan / migrate / run / compare /
locate / patch / report。不要画 PyTorch Runtime / Jittor Runtime 两个大框。不要加入云、服务器、
芯片、数据库、机器人、dashboard UI、商业插画、渐变发光、强阴影。

最终效果应该非常接近当前参考图，但做三点改进：

1. 整体背景是浅灰色论文底，不是黑色。
2. 右侧白色方法区域被清晰黑框包围，并且黑框更宽、更舒展。
3. 文字更精简、字号更大、模块内没有小段说明。
