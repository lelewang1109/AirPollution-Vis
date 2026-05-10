# Results Output and Step 5 Explanation

This folder stores generated outputs and contains the Step 5 explanation/provenance generator.

Step 5 reads Step 4 outputs and generates:

- natural-language explanation text
- uncertainty and risk statements
- provenance information
- reproducibility notes
- final JSON, Markdown, and HTML reports

The generator can use Alibaba Cloud DashScope/Qwen through the same environment key used by Step 3. If the key is unavailable or the call fails, it uses a deterministic local fallback.

## Run

Use an existing Step 4 output directory:

```bash
./gridvis-venv/bin/python modules/05_results_output/explanation_generator.py \
  --step4-output modules/05_results_output/test_step4_distribution_v2 \
  --data-source /Users/lele/Desktop/GridVis-LLM/data/2000/20000101.nc
```

Use local fallback only:

```bash
./gridvis-venv/bin/python modules/05_results_output/explanation_generator.py \
  --step4-output modules/05_results_output/test_step4_distribution_v2 \
  --data-source /Users/lele/Desktop/GridVis-LLM/data/2000/20000101.nc \
  --no-llm
```

## Python

```python
import sys

sys.path.insert(0, "modules/05_results_output")
from explanation_generator import generate_result_explanation

result = generate_result_explanation(
    "modules/05_results_output/test_step4_distribution_v2",
    data_source="/Users/lele/Desktop/GridVis-LLM/data/2000/20000101.nc",
)

print(result["explanation"])
print(result["provenance"])
```

## Outputs

By default, reports are written to:

```text
<step4-output>/step5_explanation/
```

Generated files:

- `step5_explanation.json`
- `step5_report.md`
- `step5_report.html`

The reusable prompt template is stored in:

```text
modules/05_results_output/STEP5_PROMPT.md
```
