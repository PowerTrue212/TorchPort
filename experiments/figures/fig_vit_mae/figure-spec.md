# Figure Spec

- chart_type: comparison_card
- data_sources: experiments/figures/fig_vit_mae/vit_mae.csv
- rows_in_scope: ViT and MAE DSH + TorchPort outcomes
- x_axis:
  - field: evidence
  - label: Evidence
  - unit: mixed
  - scale: linear
- y_axis:
  - field: case
  - label: Model
  - unit: category
  - scale: linear
- series_or_categories: RN, NA, SA, first divergence, weights
- legend: none
- forbidden_elements: fabricated metrics
- assumptions: ViT means vit_pytorch; MAE means mae_vit_base_patch16
