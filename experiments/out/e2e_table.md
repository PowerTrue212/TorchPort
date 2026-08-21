# SuperRes End-to-End Table

| Method | CV | RN | NA | SA | Result |
|---|---:|---:|---:|---:|---|
| Jittor official converter | ✓ | – | N/A | N/A | Generated valid code, then failed at `init.orthogonal_`. |
| DSH + TorchPort | ✓ | ✓ | ✓ | – | FP32 passed; first strict divergence was `conv3`. |

CV is code conversion, RN is runnable Jittor execution, NA is final-output numerical alignment,
and SA is stricter intermediate-layer alignment. N/A means the previous stage failed, so the metric
was not evaluated.
