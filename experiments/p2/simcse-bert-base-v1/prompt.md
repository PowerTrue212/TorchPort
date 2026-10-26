# Princeton NLP SimCSE BERT-base: native Jittor inference pilot

You are working in an independent Git checkout of the official SimCSE source.
Port the unsupervised BERT-base sentence encoder to native Jittor.

Start by checking the repository, dependencies, device, disk, data, and model
checkpoint. Implement `load_model()` in `run.py`. It must return a Jittor model.
The model takes token IDs, attention masks, and token type IDs. It returns the
last hidden state with shape `[batch, tokens, 768]`. The fixed
`evaluation_entry.py` selects the first token's hidden state. Keep that entry
unchanged.

Match the parameter names and shapes of the upstream BERT-base model. The
evaluator loads the published unsupervised SimCSE checkpoint. The task uses
fixed, pretokenized sentence pairs. The task score uses sentence-similarity
Spearman correlation with `cls_before_pooler` pooling.

You can use normal code tools, tests, network search, and the validation tools
in this session. Put temporary dependencies and caches in `.scratch/`. That
directory is excluded from delivery. When you finish, report your changes,
checks, source references, and unresolved issues.
