# Princeton NLP SimCSE BERT-base: native Jittor supervised training

You are working in an independent Git checkout of the official SimCSE source.
Port supervised SimCSE training for BERT-base to native Jittor.

Start by checking the repository, dependencies, device, disk, data, and model
files. Implement `load_train_model()`, `build_optimizer()`, and `train_step()`
in `run.py`. Keep their names and signatures.

- `load_train_model()` returns a Jittor model with the same parameter names and
  shapes as the upstream `BertForCL` with the `cls` pooler. Build it from the
  `train_config.json` in this checkout. The evaluator loads the initial state.
- `build_optimizer(model)` returns the object that `train_step` receives as
  `state`. Use AdamW with learning rate 5e-5, betas (0.9, 0.999), epsilon 1e-8,
  and weight decay 0 over all model parameters. Apply the upstream linear decay
  without warmup over 16 total steps.
- `train_step(model, state, input_ids, attention_mask, token_type_ids)` runs one
  full update and returns the scalar loss. Each input has shape `[8, 3, 32]`:
  8 NLI triplets of premise, entailment, and contradiction. Follow the upstream
  supervised objective with temperature 0.05.

Dropout is 0 in `train_config.json`. Keep it at 0. The task runs 16 fixed steps.

You can use normal code tools, tests, network search, and the validation tools
in this session. Put temporary dependencies and caches in `.scratch/`. That
directory is excluded from delivery. When you finish, report your changes,
checks, source references, and unresolved issues.
