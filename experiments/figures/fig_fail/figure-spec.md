# Figure Spec

- chart_type: bar
- data_sources: experiments/figures/fig_fail/failures.csv
- rows_in_scope: official converter failures
- x_axis:
  - field: kind
  - label: Failure reason
  - unit: category
  - scale: linear
- y_axis:
  - field: count
  - label: Cases
  - unit: count
  - scale: linear
- series_or_categories: failure kind
- legend: none
- forbidden_elements: invented failures
- assumptions: categories are derived from error-card symptoms
