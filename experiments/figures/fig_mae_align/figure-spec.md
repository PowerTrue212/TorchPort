# Figure Spec

- chart_type: heatmap_grid
- data_sources: experiments/data/tensors/vit_mae_tensors.npz
- rows_in_scope: MAE captured decoder prediction and decoder attention
- panels:
  - PyTorch reconstruction
  - Jittor reconstruction
  - PyTorch decoder attention
  - Jittor decoder attention
- layout_constraints: framework name above each image; visual type below each image
- color_mapping: identical scale for each PyTorch/Jittor pair
- forbidden_elements: PT-JT difference panel, synthetic tensors
- assumptions: reconstruction is unpatchified from decoder_pred; true mask map requires mask capture
