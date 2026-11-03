# FNO Darcy: native Jittor training migration

You are in an isolated checkout of the official NeuralOperator FNO source under `upstream/`. Implement the FNO training path in native Jittor in `evaluation_entry.py`. First inspect the repository, dependencies, device, memory, disk, bundled Darcy data, and checkpoint needs. You may use normal code tools, tests, network search, and the migration-check tools available in this session.

Provide `load_train_model()`, `build_optimizer(model)`, and `train_step(model, state, x, y)`. Match the upstream FNO with `n_modes=(4, 4)`, one input and output channel, 8 hidden channels, 2 layers, grid positional embedding, and default skip and channel MLP behavior. The evaluator loads 2,209 frozen parameters before training. It compares the scalar MSE loss and every trainable parameter after each of five fixed batches of two Darcy 16×16 samples. Use Adam with learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8, and no weight decay. Expose `param_groups` and current learning rate through the optimizer state.

The remote target has Jittor 1.3.11.0, CUDA 12.8, `conv_opt=1`, and `no_fuse=1` on an RTX 4090. You have no direct SSH access; use migration-check tools for GPU execution. The container itself is CPU-only. Keep temporary dependencies and files in `.scratch/`, excluded from delivery.

When finished, report code changes, checks, upstream references, and unresolved issues.
