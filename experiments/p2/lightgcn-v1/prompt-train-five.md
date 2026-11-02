# LightGCN LastFM: native Jittor training migration

You are in an isolated checkout containing the official LightGCN PyTorch code under
`upstream/` and an empty Jittor entry at `evaluation_entry.py`. Port the training
path to native Jittor. Start by checking the repository, dependencies, device,
memory, disk, data, and checkpoint availability. You may use normal code tools,
tests, network search, and the migration-check tools in this session.

Implement these functions in `evaluation_entry.py`:

- `load_train_model()` returns a Jittor model with trainable parameter names
  `embedding_user.weight` and `embedding_item.weight`, shapes `[1892, 64]` and
  `[4489, 64]`. Its propagation is the official three-layer LightGCN graph
  average over layers 0 through 3, using frozen LastFM train interactions.
- `build_optimizer(model)` returns the optimizer/state for Adam with learning
  rate 0.001, betas (0.9, 0.999), epsilon 1e-8, and no weight decay. The
  evaluator loads both embedding tables before building the optimizer. Expose
  `param_groups` and the current learning rate so the evaluator can inspect them.
- `train_step(model, state, users, positive_items, negative_items)` runs one
  complete BPR update and returns the scalar loss. The loss is the official
  BPR softplus term plus `1e-4` times the regularizer on the unpropagated
  embeddings, following `upstream/code/model.py` and `upstream/code/Procedure.py`.

The task uses five fixed, distinct LastFM batches of four triples each. It
loads a full checkpoint trained for ten epochs on the source implementation;
Adam moments start fresh. The validation service checks loss and full parameter
updates after every step. The remote target environment has Jittor 1.3.11.0,
CUDA 12.8, and `conv_opt=1` set for this GPU. You have no direct SSH access;
use the migration-check tools to run on the GPU. Keep temporary files and
dependencies in `.scratch/`, which is excluded from delivery.

The container is CPU-only and the validation tools provide the configured GPU.
Use those tools for GPU checks. Local dependency installation is optional; keep
it within `.scratch/` if needed.

When finished, report code changes, checks, source references, and unresolved
issues.
