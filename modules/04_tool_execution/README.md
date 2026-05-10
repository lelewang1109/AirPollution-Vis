# Tool Execution

Step 4 executes a Step 3 task graph and visualization strategy against grid data.

It performs:

- Numeric analysis: statistics, z-score anomaly detection, linear trend metrics.
- Spatial analysis: hotspot detection, connected domains, gradient strength, boundary change.
- Transformer analysis: patch tokenization, lightweight self-attention, adaptive saliency maps, and saliency overlays for generic grid or image-grid inputs.
- Rendering: PNG heatmaps, SVG charts, trajectory plots, scatter plots, and HTML visualization pages.
- Reporting: Markdown, HTML, JSON analysis results, and an execution log.

The current environment does not include `matplotlib` or `plotly`, so rendering is implemented with NumPy plus a lightweight built-in PNG/SVG writer. No extra dependency is required.

Run from the repository root:

```bash
./gridvis-venv/bin/python modules/04_tool_execution/execution.py \
  --visualization_strategy "heatmap + contour overlay" \
  --region "华北地区" \
  --variable "PM2.5" \
  --source data/2000/20000101.nc \
  --output_format png
```

Use a Step 3 plan JSON:

```bash
./gridvis-venv/bin/python modules/04_tool_execution/execution.py \
  --plan-json path/to/step3_plan.json \
  --source data/2000/20000101.nc \
  --output_format html
```

Python usage:

```python
import sys

sys.path.insert(0, "modules/04_tool_execution")
from execution import execute_plan

plan = {
    "task_graph": {
        "task_type": "distribution",
        "region": "华北地区",
        "variables": ["PM2.5"],
        "time_range": "今天",
    },
    "visualization_strategy": {
        "chart_type": "heatmap + contour overlay",
        "layout": "single map with summary cards",
        "color_encoding": "PM2.5 gradient",
    },
}

result = execute_plan(
    plan,
    "data/2000/20000101.nc",
    output_format="png",
)

print(result["output_dir"])
print(result["report_files"]["markdown"])
```

Default outputs are written to:

```text
modules/05_results_output/run_YYYYMMDD_HHMMSS/
```

Typical files:

- `heatmap_contour.png`
- `heatmap_contour.svg`
- `visualization.html`
- `transformer_saliency.png` when Transformer saliency is available
- `transformer_saliency_overlay.png` for image-grid saliency overlaid on the original RGB preview
- `analysis_results.json`
- `report.md`
- `report.html`
- `execution_log.json`
