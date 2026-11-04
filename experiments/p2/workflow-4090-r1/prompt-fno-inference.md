# FNO Darcy: native Jittor inference migration

You are in an isolated checkout of the official NeuralOperator FNO source under
`upstream/`, with an empty Jittor entry at `evaluation_entry.py`. Port the FNO inference
path to native Jittor. First inspect the repository, dependencies, device, memory, disk,
bundled Darcy data, and checkpoint needs. You may use normal code tools, tests, network
search, and the migration-check tools available in this session.

Implement `load_inference_model()` in `evaluation_entry.py`. It returns a Jittor model in
evaluation mode. Called with `x` of shape `[4, 1, 16, 16]` float32, it returns the FNO
prediction of shape `[4, 1, 16, 16]`. Match the upstream FNO with `n_modes=(4, 4)`, one
input and output channel, 8 hidden channels, 2 layers, grid positional embedding, and
default skip and channel MLP behavior. The evaluator loads a trained checkpoint of 2,209
parameters through `state_dict()` names and shapes matching upstream. Spectral weights
are complex in upstream; the checkpoint stores each complex tensor as complex values, so
your model must expose them in a form whose `state_dict()` round-trips those values.

The remote target has Jittor 1.3.11.0, CUDA 12.8, `conv_opt=1`, and `no_fuse=1` on an
RTX 4090. You have no direct SSH access; use the migration-check tools for GPU execution.
The container itself is CPU-only. Keep temporary dependencies and files in `.scratch/`,
excluded from delivery.

When finished, report code changes, checks, upstream references, and unresolved issues.
