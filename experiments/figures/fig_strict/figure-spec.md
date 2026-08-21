# Figure Spec

- chart_type: bar
- data_sources: experiments/figures/fig_strict/strict.csv
- rows_in_scope: DSH + TorchPort verified snapshot
- x_axis:
  - field: kind
  - label: Parity profile
  - unit: category
  - scale: linear
- y_axis:
  - field: count
  - label: Cases
  - unit: count
  - scale: linear
- series_or_categories: fp32 only, strict pass
- legend: none
- forbidden_elements: smoothing, extrapolation, invented baseline values
- assumptions: strict means operator-strict certificate passed
