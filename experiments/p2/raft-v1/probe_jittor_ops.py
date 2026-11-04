"""Check RAFT-critical Jittor operators in the isolated 4090 target cache."""
import json

import jittor as jt

jt.flags.use_cuda = 1
nn = jt.nn
result = {"status": "pilot_only", "jittor_version": jt.__version__, "ops": {}}
for name in ("grid_sample", "interpolate", "unfold", "InstanceNorm2d",
             "BatchNorm2d", "avg_pool2d", "softmax", "AdamW", "OneCycleLR"):
    result["ops"][name] = hasattr(nn, name)
print(json.dumps(result))
