# RAFT optical flow: native Jittor inference and training migration

You are in an isolated checkout of the official RAFT source (`princeton-vl/RAFT@2888e15`)
under `upstream/`, with an empty Jittor entry at `evaluation_entry.py`. Port both the
inference path and the training path of the standard (non-small) RAFT model to native
Jittor. First inspect the repository, dependencies, device, memory, disk, data, and
checkpoint needs. You may use normal code tools, tests, network search, and the
migration-check tools available in this session.

Implement these functions in `evaluation_entry.py`:

- `load_inference_model()` returns a Jittor model. Called with `(image1, image2)`, each
  `[2, 3, 384, 512]` float32 in `[0, 255]`, it runs 20 refinement iterations in test mode
  and returns `(flow_low, flow_up)` like upstream `RAFT.forward(..., test_mode=True)`.
- `load_train_model()` returns the same architecture in training mode.
- `build_optimizer(model)` returns the optimizer/state: AdamW, learning rate 2e-5,
  weight decay 5e-5, epsilon 1e-8, OneCycle schedule with 103 total steps, `pct_start`
  0.05, linear annealing, gradient-norm clipping at 1.0, as in `upstream/train.py`.
  Expose `param_groups` and the current learning rate.
- `train_step(model, state, image1, image2, flow_gt, valid)` runs one complete update
  with 12 iterations and the upstream sequence loss (`gamma=0.8`, `max_flow=400`) and
  returns the scalar loss.

Parameter and buffer names and shapes must match the upstream `state_dict()` of the
`raft-things.pth` checkpoint, which the evaluator loads before running. Dropout is 0 and
mixed precision is off. The evaluator checks the inference outputs, and for training the
loss and every parameter update after each of three fixed steps.

The remote target has Jittor 1.3.11.0, CUDA 12.8, and `conv_opt=1` on an RTX 4090.
Known environment issue: on this GPU, the native Jittor convolution backward can fail
with `cudaErrorIllegalAddress` for some convolution shapes. This is a framework defect,
not a migration error; a correct convolution implementation of your own avoids it.
You have no direct SSH access; use the migration-check tools for GPU execution. The
container itself is CPU-only. Keep temporary dependencies and files in `.scratch/`,
which is excluded from delivery.

When finished, report code changes, checks, upstream references, and unresolved issues.
