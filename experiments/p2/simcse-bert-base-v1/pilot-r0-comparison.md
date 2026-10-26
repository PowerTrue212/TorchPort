# SimCSE inference pilot r0: none and full

Both conditions started from seed commit
`b74039ce07b3f289a8c177ff4999c2a2d0162294`. They used the same prompt,
published checkpoint, STS-B tokens, V100, container image, proxy, and budgets.
`none` received diagnostic feedback. `full` received localized feedback.
The experiment remains `pilot_only`; the training check and formal benchmark
rules are not frozen.

| Observation | none-r0 | full-r0 |
| --- | ---: | ---: |
| Agent exit | 0 | 1 |
| Hidden inference verify | passed, 2/2 repeats | failed |
| Checkpoint state loaded | 200/200 | 199/200 |
| Missing state item | none | `model.embeddings.position_ids` |
| FP32 output comparison | passed | passed |
| STS-B source Spearman | 0.7685349386294106 | 0.7685349386294106 |
| STS-B target Spearman | 0.7685349386294106, both repeats | 0.7685349386294106, diagnostic run |
| Exploratory P_t | 100 | 100 on computed target output |
| Logical oracle queries | 2 | 3 |
| Agent wall time | 2,324.75 s | 3,685.30 s |
| Audit forbidden hits | 0 | 0 |
| Formal validity | false | false |

The full candidate produced the same STS-B score, but it failed the checkpoint
contract. Its Agent process also exited with code 1. The first controller
attempt stopped during container cleanup. A later hidden verification attempt
hit the remote filesystem quota. The controller preserved the frozen snapshot,
recovered the audit, freed regenerable `.pyc` files, and completed the hidden
verification. Raw errors remain separate from the final result.

The none candidate passed the hidden inference check. Its two target repeats
gave the same STS-B correlation. This pilot shows that the task assets, frozen
inference check, STS-B scorer, and full-repository delivery can run end to end.

Do not use these two runs to estimate a feedback effect. The full run ended
with an Agent error and faced infrastructure failures that the none run did
not. Their recorded wall times and physical costs therefore do not measure
the feedback condition alone. The formal success rate, layer diagnostic
metric, and Agent efficiency metric remain undefined for this pilot. The
training behavior also remains untested.

Detailed evidence: [none outcome](none-r0-pilot-outcome.md),
[full outcome](full-r0-pilot-outcome.md), and
[frozen tolerance](tau.json).
