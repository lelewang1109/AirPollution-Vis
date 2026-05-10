# LLM Core

Step 3 converts a user query and Step 2 grid features into a task graph and visualization strategy.

The module uses Alibaba Cloud DashScope/Qwen through the OpenAI-compatible chat completions URL:

```text
https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
```

Default model choices:

- Text planning: `qwen-plus`
- Vision/multimodal planning when image context is supplied: `qwen-vl-plus`

The API key is read from the environment. It is not printed or written to disk.

Environment variables:

- `DASHSCOPE_API_KEY`: DashScope API key.
- `ALIBABA_CLOUD_API_KEY`: fallback API key name.
- `DASHSCOPE_API_URL`: optional compatible-mode chat completions URL override.
- `DASHSCOPE_TEXT_MODEL`: optional text model override.
- `DASHSCOPE_VISION_MODEL`: optional vision model override.

Run with a raw grid source:

```bash
./gridvis-venv/bin/python modules/03_llm_core/llm_planner.py \
  "帮我看一下华北地区今天 PM2.5 分布" \
  --source data/2000/20000101.nc \
  --variables PM2.5 temp rhum
```

Run without calling the LLM, using the deterministic fallback planner:

```bash
./gridvis-venv/bin/python modules/03_llm_core/llm_planner.py \
  "近三天 PM2.5 热点区域的移动轨迹是什么？" \
  --source data/2000/20000101.nc \
  --variables PM2.5 temp rhum \
  --no-llm
```

Python usage:

```python
import sys

sys.path.insert(0, "modules/02_grid_representation")
sys.path.insert(0, "modules/03_llm_core")

from feature_extractor import extract_grid_features
from llm_planner import plan_visualization

features = extract_grid_features(
    "data/2000/20000101.nc",
    variables=["PM2.5", "temp", "rhum"],
)

plan = plan_visualization(
    "帮我看一下华北地区今天 PM2.5 分布",
    features,
)

print(plan["task_graph"])
print(plan["visualization_strategy"])
```

Returned structure:

```python
{
  "task_graph": {...},
  "visualization_strategy": {...},
  "analysis_plan": {...},
  "uncertainty_risks": [...],
  "follow_up_questions": [...],
  "semantic_response": "...",
  "llm_metadata": {...}
}
```

If the LLM call fails or the API key is missing, the module returns a rule-based plan and marks the metadata mode accordingly.
