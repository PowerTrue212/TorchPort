# Launcher 命令示例

```bash
accelerate launch --mixed_precision fp16 -m launchpkg.train --width 8 --hidden 10
torchrun --nproc_per_node=2 -m launchpkg.train --width 8 --hidden 10
```
