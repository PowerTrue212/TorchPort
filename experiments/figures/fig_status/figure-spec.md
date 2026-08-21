# Figure Spec

- chart_type: bar
- data_sources: experiments/data/current.json
- rows_in_scope: DSH + TorchPort verified snapshot
- x_axis:
  - field: metric
  - label: Metric
  - unit: count
  - scale: linear
- y_axis:
  - field: count
  - label: Cases
  - unit: count
  - scale: linear
- series_or_categories: Verified, FP32 parity, Strict parity
- legend: none
- forbidden_elements: fabricated baseline values
- assumptions: current snapshot only includes DSH + TorchPort rows
