# Architecture Figure References

This note records visual references for the TorchPort method figure. The goal is not to copy any
paper figure, but to borrow research-figure conventions that fit our contribution.

## Diagnosis

The current drafts still look like engineering workflow diagrams. They show too many internal boxes,
icons, arrows, and runtime details. A research overview figure should instead answer three questions:

1. What is the scientific gap?
2. What is the key mechanism?
3. What evidence proves the mechanism worked?

For TorchPort, the figure should foreground:

- migration is not enough without verification;
- DSH/LLM makes repository-specific decisions;
- TorchPort provides deterministic evidence;
- first divergence localization is the main technical signal;
- repair is driven by evidence, not model-specific adapters.

## Reference Set

### Self-Refine

Source: https://arxiv.org/abs/2303.17651 and https://selfrefine.info/

Useful pattern:

- one central iterative loop;
- very few node labels;
- the same agent/model performs feedback and refinement;
- the visual message is "test-time improvement by feedback", not a tool list.

Use for TorchPort:

- keep one central loop: `Understand -> Verify -> Repair`;
- show feedback as evidence from execution, not a generic dashed arrow;
- avoid many module boxes inside the loop.

### SWE-agent

Source: https://arxiv.org/abs/2405.15793 and https://github.com/SWE-agent/SWE-agent

Useful pattern:

- an agent communicates with a computer through an interface layer;
- the interface abstracts actions and feedback;
- the figure distinguishes agent, interface, and environment.

Use for TorchPort:

- draw DSH/LLM as the decision layer;
- draw TorchPort as an evidence interface between agent and repository/runtime;
- avoid making DSH look like a fixed pipeline.

### AlphaCodium

Source: https://arxiv.org/abs/2401.08500 and https://github.com/Codium-ai/AlphaCodium

Useful pattern:

- combines a method flow with a visual explanation of why validation improves results;
- the bottom panel uses a geometric "coverage / blind spot" metaphor instead of more boxes.

Use for TorchPort:

- use one panel to show that naive conversion may preserve syntax but not semantics;
- use another panel to show that numerical evidence shrinks the unknown error region;
- avoid reproducing the full multi-stage flowchart.

### CRADLE

Source: https://www.cs.purdue.edu/homes/lintan/publications/cradle-icse19.pdf

Useful pattern:

- separates detection from localization;
- emphasizes outputs such as inconsistent models, localization maps, and bugs;
- uses hidden-state extraction to localize inconsistency sources.

Use for TorchPort:

- show PyTorch/Jittor traces as hidden states or layer outputs;
- highlight the first localized divergence, not only final output mismatch;
- make localization the visual focal point.

### NNSmith

Source: https://arxiv.org/abs/2207.13066 and https://github.com/ise-uiuc/nnsmith

Useful pattern:

- compact differential-testing schematic;
- reference backend and compiled model are compared by outputs;
- failures are categorized as conversion bugs, transformation bugs, crashes, or inconsistent results.

Use for TorchPort:

- show PyTorch as reference and Jittor as migrated target;
- compare traces with same input and same weights;
- represent failures as semantic divergence, not generic red error boxes.

### TVM

Source: https://arxiv.org/abs/1802.04799

Useful pattern:

- layered abstraction from frameworks to graph to optimized deployable module;
- few large conceptual bands instead of many implementation classes;
- framework-to-target story is instantly legible.

Use for TorchPort:

- a thin `Evidence Layer` band is better than many tool buttons;
- show source framework to target framework as an abstract migration story;
- keep implementation names subordinate to the method mechanism.

### AutoGen and RepairAgent

Sources:

- https://arxiv.org/abs/2308.08155
- https://arxiv.org/abs/2403.17134
- https://github.com/sola-st/RepairAgent

Useful pattern:

- agent loops can include code execution and test feedback;
- autonomous repair is often shown as a loop around failing tests.

Use carefully:

- these figures are useful as conceptual neighbors, but they are often too engineering-heavy;
- do not imitate their many conversation/tool boxes;
- use them mainly to justify that TorchPort should show "agent + feedback + repair".

## Recommended Figure Structure

Use a three-panel paper figure instead of one large system diagram.

### Panel A: Problem Gap

Title: `Migration Is Not Verification`

Visual:

- left: PyTorch repo;
- middle: naive converter;
- right: Jittor code with a question mark over semantic behavior;
- one short note: `syntax passes, semantics unknown`.

Purpose:

- immediately explains why official conversion alone is insufficient.

### Panel B: Method

Title: `Evidence-Guided Agentic Migration`

Visual:

- DSH/LLM as a small controller above a circular loop;
- loop nodes: `Analyze`, `Verify`, `Repair`;
- a thin base layer: `TorchPort Evidence`;
- base layer emits `entry`, `trace`, `diff`, `patch`.

Purpose:

- explains that DSH decides and TorchPort grounds decisions with deterministic evidence.

### Panel C: Evidence

Title: `First Divergence Certificate`

Visual:

- two aligned layer traces: `PyTorch` and `Jittor`;
- most layer points overlap;
- one vertical orange marker: `first divergence`;
- after repair, a small green aligned trace or check mark: `aligned`.

Purpose:

- makes the core contribution concrete and measurable.

## Style Rules

- Prefer three clean panels labeled `A`, `B`, `C`.
- Use large conceptual shapes, not UI icons.
- Use at most 18 visible text labels.
- Use 1-3 words per label.
- Use a white background.
- Use thin black/gray lines for neutral structure.
- Use blue for the agent, teal for evidence, orange for divergence, green for verified output.
- Do not use heavy rounded cards, shadows, gradients, decorative icons, or logos.
- Do not number every workflow step.
- Do not list internal tool names.
