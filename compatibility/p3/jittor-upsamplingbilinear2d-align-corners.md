# Jittor `nn.UpsamplingBilinear2d` 默认 `align_corners=False`，与 PyTorch 相反

```json
{
  "schema_version": 1,
  "issue_id": "jittor-upsamplingbilinear2d-align-corners",
  "title": "同名类 nn.UpsamplingBilinear2d：PyTorch 固定 align_corners=True，Jittor 默认 False",
  "category": "api_semantics",
  "status": "observed",
  "trigger": {
    "source_framework": "pytorch",
    "target_framework": "jittor",
    "source_version": "2.8.0+cu128",
    "target_version": "1.3.9.14",
    "device": "any",
    "code_pattern": "nn.UpsamplingBilinear2d(scale_factor=2) translated name-for-name to jittor.nn.UpsamplingBilinear2d",
    "configuration": null
  },
  "symptom": {
    "stage": "numerical",
    "failure_code": "PARITY_FAILED",
    "summary": "Module config comparison reports align_corners source True / target False on the same module name."
  },
  "discrepancy": {
    "description": "On a (1,4,11,13) input with scale_factor=2, Jittor's UpsamplingBilinear2d differs from PyTorch's by up to 1.26; it matches PyTorch F.interpolate(align_corners=False) to 2.4e-7. Separately, Jittor's tuple scale_factor branch reads self.align_cornerss (typo), so a tuple scale_factor raises AttributeError. In PlantCamo full-r0 the 8 affected modules (fm1..fm4) are never called in the official forward, so this case has no numerical effect there (reported used=false)."
  },
  "reproduction": {
    "fixture": "compatibility/fixtures/jittor-upsamplingbilinear2d-align-corners/up_probe.py",
    "task": null,
    "instructions": [
      "python up_probe.py jt (Jittor env) prints the module's align_corners and saves its output.",
      "python up_probe.py torch (PyTorch env) compares against both align_corners settings."
    ],
    "expected_failure": "max|jt-torch| about 1.26 for align_corners=True"
  },
  "repair": {
    "description": "Construct jittor.nn.Upsample(scale_factor=..., mode='bilinear', align_corners=True) or call jittor.nn.interpolate with align_corners=True.",
    "patch": null,
    "constraints": ["Repair not yet verified in a model where the module is actually called."]
  },
  "evidence": {
    "before": null,
    "after": null,
    "protocol_digest": null,
    "verified": false
  },
  "applicability": {
    "known": ["Jittor 1.3.9.14 jittor.nn.UpsamplingBilinear2d source."],
    "unknown": ["Other Jittor versions.", "A real migration where the module is on the forward path."]
  },
  "provenance": {
    "source": "real_migration",
    "case_id": "plantcamo-full-r0"
  }
}
```
