# Case Table

| Case | Method | Status | FP32 | Strict | Weights | Tensors | First divergence | DSH decision |
|---|---|---:|---:|---:|---:|---:|---|---|
| mae | official_jittor | failed | – | – | 0/0 | 0 | none | syntax-error-parameter-requires-grad |
| pytorch_unet | official_jittor | converted | – | – | 0/0 | 0 | none | syntax-only |
| resnet_cifar10 | official_jittor | failed | – | – | 0/0 | 0 | none | syntax-error-line-35 |
| senet | official_jittor | converted | – | – | 0/0 | 0 | none | syntax-only |
| superres | official_jittor | converted | – | – | 0/0 | 0 | none | syntax-only |
| swin | official_jittor | failed | – | – | 0/0 | 0 | none | syntax-error-line-64 |
| vae | official_jittor | converted | – | – | 0/0 | 0 | none | syntax-only |
| vit_pytorch | official_jittor | failed | – | – | 0/0 | 0 | none | syntax-error-line-90 |
| pytorch_unet | dsh_torchport | verified | ✓ | – | 100/100 | 100 | inc.double_conv.1 | - |
| resnet_cifar10 | dsh_torchport | verified | ✓ | – | 97/97 | 84 | layer1.0.conv2 | - |
| senet | dsh_torchport | verified | ✓ | – | 125/125 | 142 | bn1 | - |
| mae | dsh_torchport | verified | ✓ | – | 254/254 | 332 | blocks.0.attn.qkv | fix-random-replay |
| superres | dsh_torchport | verified | ✓ | – | 8/8 | 10 | conv3 | fix-explicit-entry |
| swin | dsh_torchport | verified | ✓ | ✓ | 39/39 | 48 | none | fix-adaptive-pool |
| vit_pytorch | dsh_torchport | verified | ✓ | ✓ | 23/23 | 28 | none | fix-einops-backend |
| vae | dsh_torchport | verified | ✓ | – | 10/10 | 12 | fc3 | fix-source-scope |
