# LightGCN LastFM: native Jittor inference migration

You are in an isolated checkout containing the official LightGCN PyTorch code under
`upstream/` and an empty Jittor entry at `evaluation_entry.py`. Port the inference path
to native Jittor. Start by checking the repository, dependencies, device, memory, disk,
data, and checkpoint availability. You may use normal code tools, tests, network search,
and the migration-check tools in this session.

Implement `load_inference_model()` in `evaluation_entry.py`. It returns a Jittor model
with parameter names `embedding_user.weight` and `embedding_item.weight`, shapes
`[1892, 64]` and `[4489, 64]`. Its propagation is the official three-layer LightGCN graph
average over layers 0 through 3, using the frozen LastFM train interactions. Called with
an int64 vector of user ids, it returns the ratings over all items, shape
`[len(users), 4489]`, like upstream `LightGCN.getUsersRating` in `upstream/code/model.py`.
The evaluator loads a trained checkpoint into both embedding tables before running.

The remote target environment has Jittor 1.3.11.0, CUDA 12.8, and `conv_opt=1` set for
this GPU. You have no direct SSH access; use the migration-check tools to run on the GPU.
The container is CPU-only. Keep temporary files and dependencies in `.scratch/`, which is
excluded from delivery.

When finished, report code changes, checks, source references, and unresolved issues.
