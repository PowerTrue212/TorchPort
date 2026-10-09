"""在真实 PlantCamo 两端上跑 module_config + compare_module_config，确认能直接报出 eps 写反。

用法：python cfg_probe.py source <out.json> | python cfg_probe.py target <repo> <out.json> | python cfg_probe.py cmp a b
"""

import json
import sys

sys.path.insert(0, "/root/autodl-tmp/tp-work/pc-v3/src")

mode = sys.argv[1]
if mode == "source":
    import torch

    sys.path.insert(0, "/root/autodl-tmp/tp-work/1bf6a0b5967e/source")
    real = torch.load
    torch.load = lambda p, *a, **k: {} if str(p).endswith("pvt_v2_b2.pth") else real(p, *a, **k)
    from backbone.PLCamo import MyNet  # noqa: E402

    model = MyNet()  # 构造时读取硬编码 backbone 路径，补丁必须在构造之后才恢复
    torch.load = real
    from torchport.capture import module_config  # noqa: E402

    json.dump(module_config(model), open(sys.argv[2], "w"))
elif mode == "target":
    import jittor as jt  # noqa: F401

    sys.path.insert(0, sys.argv[2])
    from submission import load_model  # noqa: E402
    from torchport.capture import module_config  # noqa: E402

    json.dump(module_config(load_model()), open(sys.argv[3], "w"))
else:
    from torchport.parity import compare_module_config

    report = compare_module_config(json.load(open(sys.argv[2])), json.load(open(sys.argv[3])))
    print("compared_keys", report["compared_keys"], "common_modules", report["common_modules"],
          "source_only", report["source_only_modules"], "mismatches", len(report["mismatches"]))
    for item in report["mismatches"][:20]:
        print(" ", item["module"], item["key"], "source", item["source"], "target", item["target"])
