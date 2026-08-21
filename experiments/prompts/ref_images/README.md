# Reference Images

These images are visual references for the TorchPort method figure. They are for design study only.
Do not copy layouts verbatim.

## Recommended References

1. `swe_agent_fig1.png`
   - Source: SWE-agent, NeurIPS 2024.
   - Use: agent / interface / environment boundary.
   - TorchPort mapping: `DSH / LLM` talks to `TorchPort Evidence`, which controls repository and runtime feedback.

2. `nnsmith_fig2.png`
   - Source: NNSmith, ASPLOS 2023.
   - Use: differential testing pipeline and failure categories.
   - TorchPort mapping: PyTorch is the reference path; Jittor is the migrated target path; failures appear as semantic divergence.

3. `cradle_fig3.png`
   - Source: CRADLE, ICSE 2019.
   - Use: detection phase separated from localization phase.
   - TorchPort mapping: output mismatch is detection; first divergence is localization.

4. `self_refine_fig2.png`
   - Source: Self-Refine project page.
   - Use: compact feedback-refine-iterate loop.
   - TorchPort mapping: DSH uses numerical evidence as feedback, then edits and reruns.

5. `alphacodium_iterations.png`
   - Source: AlphaCodium repository.
   - Use: tests reduce blind spots.
   - TorchPort mapping: layer traces reduce the unknown semantic error region.

6. `alphacodium_flow.png`
   - Source: AlphaCodium repository.
   - Use: validation-driven coding pipeline.
   - TorchPort mapping: useful conceptually, but avoid copying the box-heavy style.

7. `tvm_fig2.png`
   - Source: TVM paper.
   - Use: large abstraction layers.
   - TorchPort mapping: show TorchPort as an evidence layer, not as many internal tools.

## Strongest Style Direction

Use `nnsmith_fig2.png` plus `cradle_fig3.png` for the technical story, and `swe_agent_fig1.png`
for the DSH/LLM boundary. Avoid copying AlphaCodium's dense boxes or Self-Refine's example-specific
content.
