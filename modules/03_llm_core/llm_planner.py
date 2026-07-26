from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULES_DIR = PROJECT_ROOT / "modules"
GRID_REPRESENTATION_DIR = MODULES_DIR / "02_grid_representation"
if str(GRID_REPRESENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_REPRESENTATION_DIR))

DEFAULT_DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_TEXT_MODEL = "qwen-plus"
DEFAULT_VISION_MODEL = "qwen-vl-plus"

TASK_TYPES = (
    "distribution",
    "comparison",
    "trend_analysis",
    "source_tracing",
    "clustering",
    "anomaly_detection",
    "attribution",
    "correlation_analysis",
    "summary",
)

VARIABLE_ALIASES = {
    "pm2.5": "PM2.5",
    "pm25": "PM2.5",
    "pm 2.5": "PM2.5",
    "细颗粒物": "PM2.5",
    "污染": "PM2.5",
    "temperature": "temp",
    "temp": "temp",
    "气温": "temp",
    "温度": "temp",
    "humidity": "rhum",
    "rhum": "rhum",
    "湿度": "rhum",
    "相对湿度": "rhum",
    "wind": "wind",
    "风": "wind",
    "风速": "wind",
    "precipitation": "prec",
    "prec": "prec",
    "降水": "prec",
    "降雨": "rain",
    "pressure": "pres",
    "气压": "pres",
    "image": "intensity",
    "img": "intensity",
    "picture": "intensity",
    "图像": "intensity",
    "图片": "intensity",
    "像素": "intensity",
    "亮度": "intensity",
    "灰度": "intensity",
    "边缘": "intensity",
    "纹理": "intensity",
    "red": "red",
    "green": "green",
    "blue": "blue",
    "红色": "red",
    "绿色": "green",
    "蓝色": "blue",
}

REGION_ALIASES = {
    "华北": "华北地区",
    "京津冀": "京津冀",
    "中国": "中国全境",
    "全国": "中国全境",
    "东北": "东北地区",
    "华东": "华东地区",
    "华南": "华南地区",
    "西南": "西南地区",
    "西北": "西北地区",
    "长三角": "长三角",
    "珠三角": "珠三角",
}


@dataclass
class DashScopeConfig:
    api_key_env: str = "DASHSCOPE_API_KEY"
    api_url: str = DEFAULT_DASHSCOPE_URL
    text_model: str = DEFAULT_TEXT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    timeout_seconds: int = 60
    temperature: float = 0.2
    max_tokens: int = 1800

    @classmethod
    def from_env(cls) -> "DashScopeConfig":
        return cls(
            api_key_env=os.getenv("GRIDVIS_DASHSCOPE_KEY_ENV", "DASHSCOPE_API_KEY"),
            api_url=os.getenv("DASHSCOPE_API_URL", DEFAULT_DASHSCOPE_URL),
            text_model=os.getenv("DASHSCOPE_TEXT_MODEL", DEFAULT_TEXT_MODEL),
            vision_model=os.getenv("DASHSCOPE_VISION_MODEL", DEFAULT_VISION_MODEL),
            timeout_seconds=int(os.getenv("DASHSCOPE_TIMEOUT_SECONDS", "60")),
            temperature=float(os.getenv("DASHSCOPE_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("DASHSCOPE_MAX_TOKENS", "1800")),
        )

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) or os.getenv("ALIBABA_CLOUD_API_KEY")


def plan_visualization(
    user_query: str,
    grid_features: Mapping[str, Any] | None = None,
    *,
    image_context: str | None = None,
    use_llm: bool = True,
    config: DashScopeConfig | None = None,
) -> dict[str, Any]:
    config = config or DashScopeConfig.from_env()
    compact_features = compact_grid_features(grid_features or {})
    fallback = heuristic_plan(user_query, compact_features)

    if not use_llm:
        fallback["llm_metadata"]["mode"] = "heuristic_only"
        return fallback

    if not config.api_key:
        fallback["llm_metadata"]["mode"] = "heuristic_no_api_key"
        fallback["uncertainty_risks"].append(
            "未检测到 DashScope API Key，已使用本地规则生成策略。"
        )
        return fallback

    messages = build_messages(user_query, compact_features, fallback, image_context=image_context)
    model = choose_model(user_query, image_context=image_context, config=config)

    try:
        raw = call_dashscope_chat(messages, model=model, config=config)
        llm_plan = parse_llm_json(raw)
        merged = normalize_plan(llm_plan, fallback=fallback)
        merged["llm_metadata"] = {
            "mode": "dashscope",
            "provider": "aliyun_dashscope",
            "model": model,
            "api_url": config.api_url,
            "fallback_used": False,
        }
        return merged
    except Exception as exc:
        fallback["llm_metadata"]["mode"] = "heuristic_after_llm_error"
        fallback["llm_metadata"]["provider"] = "aliyun_dashscope"
        fallback["llm_metadata"]["model"] = model
        fallback["llm_metadata"]["fallback_used"] = True
        fallback["llm_metadata"]["error"] = f"{type(exc).__name__}: {exc}"
        fallback["uncertainty_risks"].append(
            "LLM 调用或 JSON 解析失败，已使用本地规则兜底生成策略。"
        )
        return fallback


def call_dashscope_chat(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str,
    config: DashScopeConfig,
) -> str:
    api_key = config.api_key
    if not api_key:
        raise RuntimeError(f"Missing API key in {config.api_key_env} or ALIBABA_CLOUD_API_KEY.")

    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        config.api_url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DashScope HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DashScope connection error: {exc.reason}") from exc

    decoded = json.loads(response_body)
    choices = decoded.get("choices") or []
    if not choices:
        raise RuntimeError(f"DashScope returned no choices after {time.time() - started:.2f}s.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(str(part.get("text", part)) for part in content)
    if not isinstance(content, str):
        raise RuntimeError("DashScope response content is not text.")
    return content


def build_messages(
    user_query: str,
    compact_features: Mapping[str, Any],
    fallback: Mapping[str, Any],
    *,
    image_context: str | None,
) -> list[dict[str, Any]]:
    schema = {
        "task_graph": {
            "task_type": "one of: " + ", ".join(TASK_TYPES),
            "region": "目标区域",
            "variables": ["变量名"],
            "time_range": "时间范围",
            "analysis_goals": ["分析目标"],
            "constraints": ["数据/解释约束"],
        },
        "visualization_strategy": {
            "chart_type": "图表类型",
            "layout": "布局",
            "color_encoding": "颜色映射",
            "focus_region": "重点区域",
            "interaction": ["交互方式"],
            "analysis_steps": ["分析步骤"],
            "explanation_template": "面向用户的解释模板",
        },
        "analysis_plan": {
            "required_features": ["使用哪些 Step 2 特征"],
            "derived_products": ["需要派生的图层或指标"],
            "priority": "high|medium|low",
        },
        "uncertainty_risks": ["风险提示"],
        "follow_up_questions": ["必要时的问题"],
    }
    system = (
        "你是网格环境数据可视化规划助手。"
        "你的任务是把用户自然语言需求转换为结构化任务图谱和可视化策略。"
        "只能输出合法 JSON，不要输出 Markdown。"
        "不要编造数据中没有的数值；如果依据不足，要写入 uncertainty_risks。"
        "优先使用 Step 2 特征中的变量、时间、空间、缺失率、热点、趋势和相关性。"
    )
    user_content = {
        "user_query": user_query,
        "image_context": image_context,
        "grid_features": compact_features,
        "fallback_reference": compact_plan_for_prompt(fallback),
        "required_schema": schema,
        "strategy_rules": [
            "单变量单时刻优先: heatmap, choropleth, contour",
            "多时刻优先: time-slider heatmap, small multiples, pixel trend plot, spatiotemporal cube",
            "多变量优先: primary map with variable profiles, bivariate map, coupling scatter",
            "异常检测优先: anomaly mask, difference map, z-score significance map",
            "溯源或传播优先: flow overlay, trajectory map, causal explanation cards, time replay animation",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
    ]


def choose_model(
    user_query: str,
    *,
    image_context: str | None,
    config: DashScopeConfig,
) -> str:
    if image_context:
        return config.vision_model
    if any(token in user_query for token in ("图片", "截图", "图像", "看图", "多模态")):
        return config.vision_model
    return config.text_model


def parse_llm_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object.")
    return parsed


def normalize_plan(plan: Mapping[str, Any], *, fallback: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        "task_graph": merge_dict(fallback["task_graph"], plan.get("task_graph", {})),
        "visualization_strategy": merge_dict(
            fallback["visualization_strategy"],
            plan.get("visualization_strategy", {}),
        ),
        "analysis_plan": merge_dict(fallback["analysis_plan"], plan.get("analysis_plan", {})),
        "uncertainty_risks": merge_list(
            fallback.get("uncertainty_risks", []),
            plan.get("uncertainty_risks", []),
        ),
        "follow_up_questions": merge_list(
            fallback.get("follow_up_questions", []),
            plan.get("follow_up_questions", []),
        ),
        "semantic_response": plan.get("semantic_response") or fallback.get("semantic_response"),
        "llm_metadata": dict(fallback.get("llm_metadata", {})),
    }

    task_type = normalized["task_graph"].get("task_type")
    if task_type not in TASK_TYPES:
        normalized["task_graph"]["task_type"] = fallback["task_graph"]["task_type"]
    ensure_strategy_defaults(normalized)
    return normalized


def heuristic_plan(user_query: str, compact_features: Mapping[str, Any]) -> dict[str, Any]:
    variables = extract_variables(user_query, compact_features)
    region = extract_region(user_query)
    time_range = extract_time_range(user_query, compact_features)
    task_type = infer_task_type(user_query)
    chart_type = choose_chart_type(task_type, compact_features, variables)
    layout = choose_layout(task_type, compact_features, variables)
    color_encoding = choose_color_encoding(task_type, variables)
    focus_region = choose_focus_region(task_type, region)
    risks = build_uncertainty_risks(compact_features, variables)

    task_graph = {
        "task_type": task_type,
        "region": region,
        "variables": variables,
        "time_range": time_range,
        "analysis_goals": infer_analysis_goals(task_type, variables),
        "constraints": infer_constraints(compact_features),
    }
    visualization_strategy = {
        "chart_type": chart_type,
        "layout": layout,
        "color_encoding": color_encoding,
        "focus_region": focus_region,
        "interaction": infer_interactions(task_type, compact_features),
        "analysis_steps": infer_analysis_steps(task_type, variables),
        "explanation_template": build_explanation_template(task_type, variables, region, time_range),
    }
    analysis_plan = {
        "required_features": required_features_for_task(task_type, compact_features),
        "derived_products": derived_products_for_task(task_type, compact_features),
        "priority": infer_priority(task_type, compact_features),
        "transformer_ops": infer_transformer_ops(task_type, compact_features),
    }

    return {
        "task_graph": task_graph,
        "visualization_strategy": visualization_strategy,
        "analysis_plan": analysis_plan,
        "uncertainty_risks": risks,
        "follow_up_questions": infer_follow_up_questions(user_query, compact_features),
        "semantic_response": build_semantic_response(task_graph, visualization_strategy, risks),
        "llm_metadata": {
            "mode": "heuristic",
            "provider": None,
            "model": None,
            "fallback_used": True,
        },
    }


def compact_grid_features(features: Mapping[str, Any]) -> dict[str, Any]:
    global_stats = features.get("global_statistics", {}) if isinstance(features, Mapping) else {}
    spatial = features.get("spatial_features", {}) if isinstance(features, Mapping) else {}
    temporal = features.get("temporal_features", {}) if isinstance(features, Mapping) else {}
    image = features.get("image_features", {}) if isinstance(features, Mapping) else {}
    transformer = features.get("transformer_features", {}) if isinstance(features, Mapping) else {}
    semantic = features.get("semantic_summary", {}) if isinstance(features, Mapping) else {}
    metadata = features.get("metadata", {}) if isinstance(features, Mapping) else {}

    variables = list((global_stats.get("variables") or {}).keys())
    if not variables and "primary_variable" in global_stats:
        variables = [global_stats["primary_variable"]]

    variable_stats = {}
    for name, stats in (global_stats.get("variables") or {}).items():
        variable_stats[name] = {
            "min": stats.get("min"),
            "max": stats.get("max"),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "missing_ratio": stats.get("missing_ratio"),
            "unit": stats.get("unit"),
        }

    by_variable_spatial = {}
    for name, item in (spatial.get("by_variable") or {}).items():
        by_variable_spatial[name] = {
            "hotspot_count": item.get("hotspot_count"),
            "gradient_strength": item.get("gradient_strength"),
            "anisotropy": item.get("anisotropy"),
            "connected_domains": item.get("connected_domains"),
            "boundary_change": item.get("boundary_change"),
            "top_hotspots": (item.get("hotspot_locations") or [])[:3],
        }

    by_variable_temporal = {}
    for name, item in (temporal.get("by_variable") or {}).items():
        by_variable_temporal[name] = {
            "trend": item.get("trend"),
            "change_points": item.get("change_points"),
            "periodicity": item.get("periodicity"),
            "volatility": item.get("volatility"),
            "series_range": item.get("series_range"),
        }

    tensor_shape = metadata.get("tensor_shape") or []
    return json_ready(
        {
            "variables": variables,
            "primary_variable": global_stats.get("primary_variable") or semantic.get("primary_variable"),
            "variable_stats": variable_stats,
            "correlation": global_stats.get("correlation", {}),
            "spatial": {
                "primary_variable": spatial.get("primary_variable"),
                "hotspot_count": spatial.get("hotspot_count"),
                "gradient_strength": spatial.get("gradient_strength"),
                "anisotropy": spatial.get("anisotropy"),
                "connected_domains": spatial.get("connected_domains"),
                "boundary_change": spatial.get("boundary_change"),
                "by_variable": by_variable_spatial,
            },
            "temporal": {
                "primary_variable": temporal.get("primary_variable"),
                "trend": temporal.get("trend"),
                "change_points": temporal.get("change_points"),
                "periodicity": temporal.get("periodicity"),
                "volatility": temporal.get("volatility"),
                "by_variable": by_variable_temporal,
            },
            "image": {
                "is_image_grid": image.get("is_image_grid", False),
                "image_size": image.get("image_size"),
                "channels": image.get("channels"),
                "intensity": image.get("intensity"),
                "edges": image.get("edges"),
                "texture": image.get("texture"),
            },
            "transformer": {
                "enabled": transformer.get("enabled", False),
                "model": transformer.get("model", {}),
                "saliency_summary": transformer.get("saliency_summary", {}),
                "top_salient_patches": (transformer.get("top_salient_patches") or [])[:5],
                "attention_edges": (transformer.get("attention_edges") or [])[:8],
            },
            "semantic_summary": semantic,
            "metadata": {
                "tensor_shape": tensor_shape,
                "time_steps": tensor_shape[0] if len(tensor_shape) >= 1 else None,
                "variable_count": tensor_shape[1] if len(tensor_shape) >= 2 else len(variables),
                "source_summary": metadata.get("source_summary", {}),
            },
        }
    )


def extract_variables(user_query: str, compact_features: Mapping[str, Any]) -> list[str]:
    query_lower = user_query.lower()
    available = compact_features.get("variables") or []
    detected = []
    for alias, canonical in VARIABLE_ALIASES.items():
        if alias.lower() in query_lower or alias in user_query:
            detected.append(match_available_variable(canonical, available))
    for var in available:
        if str(var).lower() in query_lower and var not in detected:
            detected.append(var)
    detected = unique_non_empty(detected)
    if detected:
        return detected
    primary = compact_features.get("primary_variable")
    if primary:
        return [primary]
    return ["PM2.5"]


def match_available_variable(candidate: str, available: Sequence[str]) -> str:
    if not available:
        return candidate
    candidate_lower = candidate.lower()
    for var in available:
        if str(var).lower() == candidate_lower:
            return str(var)
    for var in available:
        if candidate_lower in str(var).lower() or str(var).lower() in candidate_lower:
            return str(var)
    return candidate


def extract_region(user_query: str) -> str:
    for alias, canonical in REGION_ALIASES.items():
        if alias in user_query:
            return canonical
    city_match = re.search(r"([\u4e00-\u9fa5]{2,8})(市|省|区|县)", user_query)
    if city_match:
        region = "".join(city_match.groups())
        if region not in {"热点区", "热点区域", "异常区", "异常区域", "目标区", "目标区域"}:
            return region
    return "数据覆盖区域"


def extract_time_range(user_query: str, compact_features: Mapping[str, Any]) -> str:
    patterns = [
        (r"今天.*昨天|昨天.*今天", "今天与昨天"),
        (r"近\s*(\d+)\s*天", lambda m: f"近{m.group(1)}天"),
        (r"过去\s*(\d+)\s*天", lambda m: f"过去{m.group(1)}天"),
        (r"近\s*一\s*周|过去\s*一\s*周|最近\s*一\s*周", "近一周"),
        (r"近\s*三\s*天|过去\s*三\s*天", "近三天"),
        (r"今天|今日", "今天"),
        (r"昨天|昨日", "昨天"),
        (r"\d{4}年\d{1,2}月", lambda m: m.group(0)),
        (r"\d{4}-\d{1,2}-\d{1,2}", lambda m: m.group(0)),
    ]
    for pattern, value in patterns:
        match = re.search(pattern, user_query)
        if match:
            return value(match) if callable(value) else value

    source_summary = compact_features.get("metadata", {}).get("source_summary", {})
    time_coverage = (
        source_summary.get("spatiotemporal_summary", {}).get("time_coverage", {})
        if isinstance(source_summary, Mapping)
        else {}
    )
    start = time_coverage.get("start")
    end = time_coverage.get("end")
    if start and end:
        return start if start == end else f"{start} 至 {end}"
    return "数据时间范围"


def infer_task_type(user_query: str) -> str:
    q = user_query.lower()
    if any(token in q or token in user_query for token in ("比较", "对比", "差异", "相比", "昨天")):
        return "comparison"
    if any(token in q or token in user_query for token in ("趋势", "变化", "演化", "过去", "近", "移动轨迹", "回放")):
        if any(token in user_query for token in ("轨迹", "来源", "扩散", "传播", "溯源")):
            return "source_tracing"
        return "trend_analysis"
    if any(token in user_query for token in ("来源", "溯源", "扩散", "传播", "输送")):
        return "source_tracing"
    if any(token in user_query for token in ("聚类", "分群", "分区")):
        return "clustering"
    if any(token in user_query for token in ("异常", "突变", "极端", "显著")):
        return "anomaly_detection"
    if any(token in user_query for token in ("边缘", "轮廓", "纹理", "图像", "图片", "像素", "亮度", "灰度")):
        return "distribution"
    if any(token in user_query for token in ("影响", "归因", "关系", "相关", "导致")):
        return "attribution"
    if any(token in user_query for token in ("分布", "看一下", "展示", "地图")):
        return "distribution"
    return "summary"


def choose_chart_type(task_type: str, compact_features: Mapping[str, Any], variables: Sequence[str]) -> str:
    time_steps = compact_features.get("metadata", {}).get("time_steps") or 1
    variable_count = len(variables)
    if is_image_features(compact_features):
        if task_type == "anomaly_detection":
            return "image anomaly mask + edge significance map"
        if task_type == "comparison":
            return "side-by-side image raster + difference map"
        return "image raster + intensity heatmap + edge/texture overlay"
    if task_type == "anomaly_detection":
        return "anomaly mask + z-score significance map"
    if task_type == "comparison":
        return "side-by-side heatmap + difference map"
    if task_type == "source_tracing":
        return "trajectory map + flow overlay + time replay animation"
    if task_type == "clustering":
        return "clustered hotspot map"
    if task_type in {"attribution", "correlation_analysis"}:
        return "primary map + variable coupling scatter"
    if time_steps and time_steps > 1:
        return "time-slider heatmap + small multiples"
    if variable_count > 1:
        return "primary variable heatmap + linked variable profiles"
    return "heatmap + contour overlay"


def choose_layout(task_type: str, compact_features: Mapping[str, Any], variables: Sequence[str]) -> str:
    time_steps = compact_features.get("metadata", {}).get("time_steps") or 1
    if is_image_features(compact_features):
        return "image canvas with channel, histogram, and texture panels"
    if task_type in {"comparison", "attribution", "source_tracing"}:
        return "multi-panel dashboard"
    if time_steps > 1:
        return "map on top, timeline and small multiples below"
    if len(variables) > 1:
        return "main map with right-side variable relationship panel"
    return "single map with summary cards"


def choose_color_encoding(task_type: str, variables: Sequence[str]) -> str:
    primary = variables[0] if variables else "value"
    if primary in {"intensity", "red", "green", "blue", "alpha"}:
        return f"native image colors plus grayscale/sequential scale for {primary}; edges encoded by overlay opacity"
    if task_type == "anomaly_detection":
        return "diverging z-score scale centered at 0; significant cells highlighted"
    if task_type == "comparison":
        return f"sequential scale for {primary}; diverging scale for difference"
    if task_type in {"attribution", "correlation_analysis"} and len(variables) > 1:
        return f"sequential scale for {primary}; secondary variable encoded by scatter color/shape"
    if task_type == "source_tracing":
        return f"sequential scale for {primary}; trajectory age encoded by line opacity"
    return f"sequential perceptual color scale for {primary}, with high values in darker tones"


def choose_focus_region(task_type: str, region: str) -> str:
    if task_type in {"clustering", "source_tracing"}:
        return f"{region}的热点区域"
    if task_type == "anomaly_detection":
        return f"{region}的异常高值与突变区域"
    return region


def infer_analysis_goals(task_type: str, variables: Sequence[str]) -> list[str]:
    primary = variables[0] if variables else "目标变量"
    if primary in {"intensity", "red", "green", "blue", "alpha"}:
        return ["识别图像网格的亮度分布、边缘结构和纹理复杂度", "定位高亮、低亮或局部变化显著的像素区域"]
    goals = {
        "distribution": [f"识别{primary}空间分布格局", "定位高值和低值区域"],
        "comparison": [f"比较{primary}在不同时间或区域的差异", "突出变化幅度最大的区域"],
        "trend_analysis": [f"识别{primary}时间趋势", "检测突变点和周期性"],
        "source_tracing": [f"追踪{primary}热点移动路径", "结合风场或相关变量解释传播方向"],
        "clustering": [f"将{primary}热点区域聚类", "比较不同热点簇的强度和范围"],
        "anomaly_detection": [f"检测{primary}异常区域", "输出异常掩膜和显著性说明"],
        "attribution": [f"解释其他变量对{primary}的影响", "量化变量关系与不确定性"],
        "correlation_analysis": [f"分析{primary}与其他变量的相关性"],
        "summary": [f"概括{primary}的主要空间和时间特征"],
    }
    return goals.get(task_type, goals["summary"])


def infer_constraints(compact_features: Mapping[str, Any]) -> list[str]:
    constraints = []
    if is_image_features(compact_features):
        constraints.append("该输入是图像网格，空间坐标表示像素索引，不应解释为经纬度或行政区域。")
    time_steps = compact_features.get("metadata", {}).get("time_steps")
    if time_steps == 1:
        constraints.append("数据只有单个时间片，不能直接支持趋势或传播速度结论。")
    for name, stats in (compact_features.get("variable_stats") or {}).items():
        missing = stats.get("missing_ratio")
        if missing is not None and missing > 0.3:
            constraints.append(f"{name} 缺失率较高：{missing:.1%}，解释需结合有效区域掩膜。")
    if not constraints:
        constraints.append("优先基于有效区域和 Step 2 特征解释，不外推到数据覆盖范围外。")
    return constraints


def infer_interactions(task_type: str, compact_features: Mapping[str, Any]) -> list[str]:
    interactions = ["hover cell value and coordinates", "toggle valid-region mask"]
    time_steps = compact_features.get("metadata", {}).get("time_steps") or 1
    if is_image_features(compact_features):
        interactions.extend(["toggle RGB/intensity view", "toggle edge and texture overlays", "inspect pixel channel values"])
    if time_steps > 1:
        interactions.append("time slider")
    if task_type in {"comparison", "anomaly_detection"}:
        interactions.append("brush high-change cells")
    if task_type in {"source_tracing", "trend_analysis"}:
        interactions.append("playback animation")
    if task_type in {"attribution", "correlation_analysis"}:
        interactions.append("linked map-scatter brushing")
    return interactions


def infer_analysis_steps(task_type: str, variables: Sequence[str]) -> list[str]:
    primary = variables[0] if variables else "目标变量"
    if primary in {"intensity", "red", "green", "blue", "alpha"}:
        return ["渲染原始图像网格", "计算亮度分布、边缘强度和局部纹理", "标注高梯度或高纹理区域并生成解释"]
    steps = {
        "distribution": [f"绘制{primary}有效区域热力图", "叠加热点边界", "生成空间格局解释"],
        "comparison": ["对齐两个时间片或区域", "计算差值和变化率", "突出变化最大的连通区域"],
        "trend_analysis": ["计算空间平均时间序列", "估计趋势斜率", "检测突变点和周期性"],
        "source_tracing": ["提取热点质心序列", "估计移动方向", "结合风速/风向或相关变量解释来源"],
        "clustering": ["提取热点连通域", "基于强度和位置聚类", "输出每个簇的摘要"],
        "anomaly_detection": ["计算基线均值和标准差", "生成 z-score 图层", "输出异常掩膜和风险提示"],
        "attribution": ["计算变量相关性", "绘制耦合散点", "结合空间共现解释可能影响"],
        "correlation_analysis": ["计算变量相关性", "对高相关区域做局部解释"],
        "summary": ["汇总全局统计", "提取热点和趋势", "生成面向用户的简短结论"],
    }
    return steps.get(task_type, steps["summary"])


def required_features_for_task(task_type: str, compact_features: Mapping[str, Any] | None = None) -> list[str]:
    base = ["global_statistics", "spatial_features"]
    # Image grids still use spatial statistics, but the planner should also
    # preserve image-specific edge, texture, and channel summaries when present.
    if compact_features and is_image_features(compact_features):
        base.append("image_features")
    if compact_features and (compact_features.get("transformer") or {}).get("enabled"):
        base.append("transformer_features")
    if task_type in {"trend_analysis", "source_tracing", "comparison"}:
        base.append("temporal_features")
    if task_type in {"attribution", "correlation_analysis"}:
        base.append("global_statistics.correlation")
    if task_type in {"clustering", "source_tracing"}:
        base.append("embeddings.block_embedding")
    return unique_non_empty(base)


def derived_products_for_task(task_type: str, compact_features: Mapping[str, Any] | None = None) -> list[str]:
    if compact_features and is_image_features(compact_features):
        return [
            "RGB image preview",
            "intensity heatmap",
            "edge-strength map",
            "texture summary",
            "channel histogram",
            "transformer patch saliency map",
        ]
    products = {
        "distribution": ["valid-region heatmap", "hotspot boundary layer"],
        "comparison": ["difference grid", "change-rate grid", "high-change mask"],
        "trend_analysis": ["spatial mean time series", "trend slope map", "change-point markers"],
        "source_tracing": ["hotspot centroid trajectory", "flow overlay", "causal explanation card"],
        "clustering": ["hotspot cluster labels", "cluster summary cards"],
        "anomaly_detection": ["z-score grid", "anomaly mask", "uncertainty mask"],
        "attribution": ["variable coupling scatter", "correlation matrix", "co-location map"],
        "correlation_analysis": ["correlation matrix", "linked scatter-map view"],
        "summary": ["summary cards", "key hotspot table"],
    }
    return products.get(task_type, products["summary"])


def is_image_features(compact_features: Mapping[str, Any]) -> bool:
    return bool((compact_features.get("image") or {}).get("is_image_grid"))


def infer_transformer_ops(task_type: str, compact_features: Mapping[str, Any]) -> list[str]:
    if not (compact_features.get("transformer") or {}).get("enabled"):
        return []
    ops = ["patch_tokenization", "2d_positional_encoding", "multi_head_self_attention", "patch_saliency"]
    if is_image_features(compact_features):
        ops.append("image_text_alignment_ready")
    if task_type in {"anomaly_detection", "clustering"}:
        ops.append("saliency_guided_region_selection")
    return ops


def infer_priority(task_type: str, compact_features: Mapping[str, Any]) -> str:
    if task_type in {"anomaly_detection", "source_tracing"}:
        return "high"
    hotspot_count = compact_features.get("spatial", {}).get("hotspot_count")
    if isinstance(hotspot_count, (int, float)) and hotspot_count > 0:
        return "medium"
    return "low"


def build_uncertainty_risks(compact_features: Mapping[str, Any], variables: Sequence[str]) -> list[str]:
    risks = []
    variable_stats = compact_features.get("variable_stats") or {}
    for var in variables:
        stats = variable_stats.get(var, {})
        missing = stats.get("missing_ratio")
        if missing is not None and missing > 0.3:
            risks.append(f"{var} 原始缺失率约 {missing:.1%}，固定缺失或区域外填补可能影响边界解读。")
    time_steps = compact_features.get("metadata", {}).get("time_steps")
    if time_steps == 1:
        if is_image_features(compact_features):
            risks.append("当前输入是单帧图像网格，不支持时序趋势、运动轨迹或因果方向结论。")
        else:
            risks.append("当前特征只有单个时间片，趋势、移动轨迹和因果方向只能作为待验证假设。")
    if not risks:
        risks.append("结论应限定在数据覆盖范围和有效区域内。")
    return risks


def infer_follow_up_questions(user_query: str, compact_features: Mapping[str, Any]) -> list[str]:
    questions = []
    task_type = infer_task_type(user_query)
    time_steps = compact_features.get("metadata", {}).get("time_steps")
    if is_image_features(compact_features):
        questions.append("是否需要框选图像中的局部区域，单独分析边缘、纹理或异常像素？")
        questions.append("是否需要按 RGB 通道分别比较亮度和纹理差异？")
        return questions[:3]
    if task_type in {"comparison", "trend_analysis", "source_tracing"} and time_steps == 1:
        questions.append("是否提供更多时间片，用于支持对比、趋势或传播路径分析？")
    if "区域" not in user_query and extract_region(user_query) == "数据覆盖区域":
        questions.append("是否需要限定到具体区域，例如华北、京津冀或中国全境？")
    return questions[:3]


def build_explanation_template(task_type: str, variables: Sequence[str], region: str, time_range: str) -> str:
    primary = variables[0] if variables else "目标变量"
    if primary in {"intensity", "red", "green", "blue", "alpha"}:
        return f"{region}图像网格的{primary}呈现{{亮度/通道分布}}；边缘集中在{{高梯度区域}}，纹理复杂度为{{纹理描述}}。"
    templates = {
        "distribution": f"{time_range}，{region}{primary}呈现{{空间格局}}；高值主要位于{{热点区域}}，需结合缺失掩膜判断边界可信度。",
        "comparison": f"对比{time_range}，{region}{primary}在{{变化区域}}出现{{上升/下降}}，差值峰值位于{{位置}}。",
        "trend_analysis": f"{time_range}内，{region}{primary}总体趋势为{{趋势}}；突变点出现在{{时间}}，波动性为{{波动描述}}。",
        "source_tracing": f"{time_range}，{primary}热点轨迹表现为{{移动方向}}；可能来源为{{候选来源}}，该判断需风场或排放数据进一步验证。",
        "clustering": f"{region}{primary}热点可分为{{簇数量}}类；主要簇位于{{区域}}，强度差异为{{差异描述}}。",
        "anomaly_detection": f"{region}{primary}异常区域集中在{{异常区域}}；z-score 为{{显著性}}，缺失率为{{缺失率}}。",
        "attribution": f"{region}{primary}与{{关联变量}}的关系为{{正/负相关}}；空间共现区域位于{{区域}}，不代表直接因果。",
        "correlation_analysis": f"{primary}与其他变量的最强关系为{{变量关系}}；相关区域集中在{{区域}}。",
        "summary": f"{region}{primary}的主要特征包括{{热点}}、{{趋势}}和{{风险提示}}。",
    }
    return templates.get(task_type, templates["summary"])


def build_semantic_response(
    task_graph: Mapping[str, Any],
    visualization_strategy: Mapping[str, Any],
    risks: Sequence[str],
) -> str:
    variables = ", ".join(task_graph.get("variables") or [])
    return (
        f"已将请求解析为 {task_graph.get('task_type')} 任务，目标区域为 {task_graph.get('region')}，"
        f"目标变量为 {variables}。建议使用 {visualization_strategy.get('chart_type')}，"
        f"布局为 {visualization_strategy.get('layout')}。风险提示：{risks[0] if risks else '无明显风险'}"
    )


def ensure_strategy_defaults(plan: dict[str, Any]) -> None:
    plan.setdefault("task_graph", {})
    plan.setdefault("visualization_strategy", {})
    plan.setdefault("analysis_plan", {})
    plan.setdefault("uncertainty_risks", [])
    plan.setdefault("follow_up_questions", [])

    task_graph = plan["task_graph"]
    strategy = plan["visualization_strategy"]
    task_graph.setdefault("task_type", "summary")
    task_graph.setdefault("region", "数据覆盖区域")
    task_graph.setdefault("variables", ["PM2.5"])
    task_graph.setdefault("time_range", "数据时间范围")
    task_graph.setdefault("analysis_goals", [])
    task_graph.setdefault("constraints", [])

    strategy.setdefault("chart_type", "heatmap")
    strategy.setdefault("layout", "single map with summary cards")
    strategy.setdefault("color_encoding", "sequential perceptual color scale")
    strategy.setdefault("focus_region", task_graph["region"])
    strategy.setdefault("interaction", [])
    strategy.setdefault("analysis_steps", [])
    strategy.setdefault("explanation_template", "")


def merge_dict(base: Mapping[str, Any], override: Any) -> dict[str, Any]:
    merged = dict(base)
    if isinstance(override, Mapping):
        for key, value in override.items():
            if value not in (None, "", [], {}):
                merged[key] = value
    return merged


def merge_list(base: Any, override: Any) -> list[Any]:
    items = []
    if isinstance(base, list):
        items.extend(base)
    if isinstance(override, list):
        items.extend(override)
    elif override:
        items.append(override)
    return unique_non_empty(items)


def unique_non_empty(values: Sequence[Any]) -> list[Any]:
    seen = set()
    output = []
    for value in values:
        if value in (None, ""):
            continue
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(value)
    return output


def compact_plan_for_prompt(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_graph": plan.get("task_graph"),
        "visualization_strategy": plan.get("visualization_strategy"),
        "analysis_plan": plan.get("analysis_plan"),
        "uncertainty_risks": plan.get("uncertainty_risks"),
    }


def json_ready(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def load_features_from_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_features_from_source(
    source: str | Path,
    *,
    variables: Sequence[str] | None,
) -> dict[str, Any]:
    from feature_extractor import extract_grid_features

    return extract_grid_features(source, variables=variables)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM visualization plans from grid features and user queries.")
    parser.add_argument("query", help="Natural-language user query.")
    parser.add_argument("--features-json", help="Path to a Step 2 features JSON file.")
    parser.add_argument("--source", help="Raw data source accepted by Step 1/2.")
    parser.add_argument("--variables", nargs="*", default=None, help="Variables to use when --source is supplied.")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic local planning only.")
    parser.add_argument("--image-context", help="Optional image context text; selects the vision model.")
    parser.add_argument("--model", help="Override text model name.")
    args = parser.parse_args()

    if args.features_json:
        features = load_features_from_json(args.features_json)
    elif args.source:
        features = load_features_from_source(args.source, variables=args.variables)
    else:
        features = {}

    config = DashScopeConfig.from_env()
    if args.model:
        config.text_model = args.model

    plan = plan_visualization(
        args.query,
        features,
        image_context=args.image_context,
        use_llm=not args.no_llm,
        config=config,
    )
    print(json.dumps(json_ready(plan), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
