# Figure Spec

- chart_type: heatmap
- data_sources: experiments/figures/fig_models/models.csv
- rows_in_scope: DSH + TorchPort model-level outcomes
- x_axis:
  - field: metric
  - label: Result metric
  - unit: category
  - scale: linear
- y_axis:
  - field: case
  - label: Model / repository
  - unit: category
  - scale: linear
- series_or_categories: pass/fail status
- color_mapping: pass = blue, fail = light gray
- legend: none
- forbidden_elements: fabricated model outcomes
- assumptions: status is derived from experiments/data/current.json
