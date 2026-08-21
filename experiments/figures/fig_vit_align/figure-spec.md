# Figure Spec

- chart_type: heatmap_grid
- data_sources: experiments/data/tensors/vit_mae_tensors.npz
- rows_in_scope: ViT captured attention and patch features
- panels:
  - PyTorch attention
  - Jittor attention
  - PyTorch patch feature
  - Jittor patch feature
- layout_constraints: framework name above each image; visual type below each image
- color_mapping: identical viridis scale for each PyTorch/Jittor pair
- forbidden_elements: PT-JT difference panel, synthetic tensors
- assumptions: attention is mean over heads with CLS token removed
