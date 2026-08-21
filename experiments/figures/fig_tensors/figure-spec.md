# Figure Spec

- chart_type: heatmap_grid
- data_sources: experiments/data/tensors/vit_mae_tensors.npz
- rows_in_scope: ViT and MAE captured intermediate tensors
- panels:
  - ViT attention map, PyTorch
  - ViT attention map, Jittor
  - MAE reconstruction, PyTorch
  - MAE reconstruction, Jittor
- layout_constraints: framework name above each image; visual type below each image
- color_mapping: identical scale for each PyTorch/Jittor pair
- forbidden_elements: PT-JT difference panel, synthetic activations or fabricated tensors
- assumptions: tensors are extracted from TorchPort source.npz and target.npz certificates
