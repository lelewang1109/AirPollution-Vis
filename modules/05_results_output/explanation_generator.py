from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULES_DIR = PROJECT_ROOT / "modules"
LLM_CORE_DIR = MODULES_DIR / "03_llm_core"
if str(LLM_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_CORE_DIR))

from llm_planner import DashScopeConfig, call_dashscope_chat, parse_llm_json  # noqa: E402


DEFAULT_TEXT_MODEL = "qwen-plus"


def generate_result_explanation(
    step4_output: str | Path | Mapping[str, Any],
    *,
    data_source: str | None = None,
    output_dir: str | Path | None = None,
    use_llm: bool = True,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate Step 5 explanation, provenance, uncertainty, and report files."""

    bundle = load_step4_bundle(step4_output)
    evidence = build_evidence(bundle, data_source=data_source)
    fallback = build_fallback_explanation(evidence)

    if use_llm:
        llm_result = try_llm_explanation(evidence, fallback=fallback, model=model)
    else:
        llm_result = dict(fallback)
        llm_result["llm_metadata"] = {"mode": "heuristic_only", "provider": None, "model": None, "fallback_used": True}

    explanation = normalize_explanation(llm_result, fallback=fallback)
    target_dir = prepare_output_dir(output_dir, bundle)
    report_files = write_step5_reports(target_dir, explanation, evidence)
    explanation["report_files"] = report_files
    explanation["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return explanation


def load_step4_bundle(step4_output: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(step4_output, Mapping):
        analysis = dict(step4_output)
        base_dir = Path(analysis.get("output_dir", MODULES_DIR / "05_results_output"))
        log = load_json_if_exists(base_dir / "execution_log.json", default=[])
        return {"analysis": analysis, "execution_log": log, "base_dir": base_dir}

    path = Path(step4_output)
    if path.is_dir():
        analysis_path = path / "analysis_results.json"
        log_path = path / "execution_log.json"
        if not analysis_path.exists():
            raise FileNotFoundError(f"Missing Step 4 analysis file: {analysis_path}")
        return {
            "analysis": load_json_if_exists(analysis_path, default={}),
            "execution_log": load_json_if_exists(log_path, default=[]),
            "base_dir": path,
        }
    if path.name == "analysis_results.json":
        base_dir = path.parent
        return {
            "analysis": load_json_if_exists(path, default={}),
            "execution_log": load_json_if_exists(base_dir / "execution_log.json", default=[]),
            "base_dir": base_dir,
        }
    raise ValueError("step4_output must be a Step 4 output directory, analysis_results.json, or result dictionary.")


def build_evidence(bundle: Mapping[str, Any], *, data_source: str | None) -> dict[str, Any]:
    analysis = bundle["analysis"]
    plan = analysis.get("plan") or {}
    numeric = analysis.get("numeric") or analysis.get("analysis_results", {}).get("numeric", {})
    spatial = analysis.get("spatial") or analysis.get("analysis_results", {}).get("spatial", {})
    temporal = analysis.get("temporal") or analysis.get("analysis_results", {}).get("temporal", {})
    transformer = analysis.get("transformer") or analysis.get("analysis_results", {}).get("transformer", {})
    artifacts = analysis.get("artifacts", [])
    step4_explanation = analysis.get("explanation", {})
    execution_log = bundle.get("execution_log", [])
    base_dir = Path(bundle["base_dir"])

    stats = numeric.get("statistics", {})
    anomaly = numeric.get("anomaly", {})
    trend = temporal.get("trend", {})
    hotspots = spatial.get("hotspots", {})
    gradient = spatial.get("gradient", {})

    source = (
        data_source
        or find_data_source_in_plan(plan)
        or os.getenv("GRIDVIS_DATA_SOURCE")
        or "未提供数据源路径"
    )
    tools_used = infer_tools_used(execution_log, artifacts)
    process_steps = infer_process_steps(plan, execution_log)
    visualization_type = (plan.get("visualization_strategy") or {}).get("chart_type") or infer_visualization_type(artifacts)
    image_grid = is_image_evidence(visualization_type, artifacts, stats)
    risk_items = list(step4_explanation.get("risks") or [])
    if image_grid:
        risk_items = [
            item
            for item in risk_items
            if "传播路径" not in str(item) and "趋势" not in str(item)
        ]
    risk_items.extend(infer_uncertainty(stats, trend, anomaly, image_grid=image_grid))

    return json_ready(
        {
            "visualization_type": visualization_type,
            "grid_semantics": {
                "is_image_grid": image_grid,
                "coordinate_system": "pixel" if image_grid else "geographic_or_projected_or_unknown",
            },
            "task_graph": plan.get("task_graph", {}),
            "visualization_strategy": plan.get("visualization_strategy", {}),
            "data_source": source,
            "output_directory": str(base_dir),
            "artifacts": summarize_artifacts(artifacts, base_dir),
            "statistics": {
                "variable": stats.get("variable"),
                "unit": stats.get("unit"),
                "min": stats.get("min"),
                "max": stats.get("max"),
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "p95": stats.get("p95"),
                "valid_count": stats.get("valid_count"),
                "original_missing_ratio": stats.get("original_missing_ratio"),
            },
            "spatial": {
                "hotspot_count": hotspots.get("hotspot_count"),
                "connected_domains": hotspots.get("connected_domains"),
                "hotspot_threshold": hotspots.get("threshold"),
                "top_components": (hotspots.get("components") or [])[:5],
                "gradient_strength": gradient.get("gradient_strength"),
                "gradient_p95": gradient.get("gradient_p95"),
                "anisotropy": gradient.get("anisotropy"),
                "boundary_change": gradient.get("boundary_change"),
            },
            "temporal": {
                "trend": trend.get("trend"),
                "slope": trend.get("slope"),
                "r2": trend.get("r2"),
                "series": (temporal.get("series") or [])[:30],
                "change_points": temporal.get("change_points", []),
                "time": (temporal.get("time") or [])[:30],
            },
            "anomaly": {
                "method": anomaly.get("method"),
                "threshold": anomaly.get("threshold"),
                "count": anomaly.get("count"),
                "top_locations": (anomaly.get("locations") or [])[:5],
            },
            "transformer": {
                "enabled": transformer.get("enabled", False),
                "model": transformer.get("model", {}),
                "saliency_summary": transformer.get("saliency_summary", {}),
                "saliency_method": transformer.get("saliency_method", {}),
                "top_salient_patches": (transformer.get("top_salient_patches") or [])[:5],
                "attention_edges": (transformer.get("attention_edges") or [])[:8],
            },
            "uncertainty_inputs": unique_non_empty(risk_items),
            "analysis_process": process_steps,
            "calculation_parameters": infer_calculation_parameters(plan, numeric, spatial, temporal),
            "tools_used": tools_used,
            "execution_log_summary": summarize_execution_log(execution_log),
        }
    )


def try_llm_explanation(
    evidence: Mapping[str, Any],
    *,
    fallback: Mapping[str, Any],
    model: str | None,
) -> dict[str, Any]:
    config = DashScopeConfig.from_env()
    config.text_model = model or config.text_model or DEFAULT_TEXT_MODEL
    if not config.api_key:
        result = dict(fallback)
        result["llm_metadata"] = {
            "mode": "heuristic_no_api_key",
            "provider": None,
            "model": None,
            "fallback_used": True,
        }
        return result

    messages = build_llm_messages(evidence, fallback)
    try:
        raw = call_dashscope_chat(messages, model=config.text_model, config=config)
        parsed = parse_llm_json(raw)
        parsed["llm_metadata"] = {
            "mode": "dashscope",
            "provider": "aliyun_dashscope",
            "model": config.text_model,
            "fallback_used": False,
        }
        return parsed
    except Exception as exc:
        result = dict(fallback)
        result["llm_metadata"] = {
            "mode": "heuristic_after_llm_error",
            "provider": "aliyun_dashscope",
            "model": config.text_model,
            "fallback_used": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
        return result


def build_llm_messages(evidence: Mapping[str, Any], fallback: Mapping[str, Any]) -> list[dict[str, str]]:
    system = (
        "你是 GridVis-LLM 的结果解释与溯源助手。"
        "你必须基于给定证据生成中文分析解释、风险提示和可复现溯源信息。"
        "只能输出合法 JSON，不要输出 Markdown。"
        "不要编造证据中没有的数值、工具或因果结论。"
        "如果依据不足，用“不足以判断”或“需要更多数据验证”。"
    )
    schema = {
        "visualization_type": "图表类型",
        "explanation": "自然语言解释文本，覆盖趋势、热点、异常或变量影响",
        "uncertainty_risk": "不确定性和风险提示",
        "data_source": "数据来源",
        "analysis_process": "分析过程说明",
        "calculation_parameters": "计算参数与方法",
        "provenance": {
            "analysis_method": "方法链路",
            "tools_used": ["工具及用途"],
            "execution_log": "日志摘要",
            "artifacts": ["图表或报告文件"],
            "reproducibility_notes": ["复现注意事项"],
        },
        "follow_up_questions": ["建议用户继续追问的问题"],
    }
    user_payload = {
        "evidence": evidence,
        "fallback_reference": fallback,
        "required_schema": schema,
        "writing_requirements": [
            "解释要短而具体，优先引用均值、最大值、热点数量、趋势、异常数量、缺失率。",
            "将缺失率、单时间片、样本不足、填补方法等写入风险提示。",
            "工具链必须以 evidence.tools_used 和 execution_log_summary 为准。",
            "不得声称使用 pysal、matplotlib 或 plotly，除非 evidence 中明确出现。",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def build_fallback_explanation(evidence: Mapping[str, Any]) -> dict[str, Any]:
    stats = evidence.get("statistics", {})
    spatial = evidence.get("spatial", {})
    temporal = evidence.get("temporal", {})
    anomaly = evidence.get("anomaly", {})
    transformer = evidence.get("transformer", {})
    variable = stats.get("variable") or "目标变量"
    unit = stats.get("unit") or ""
    unit_text = f" {unit}" if unit else ""
    region = (evidence.get("task_graph") or {}).get("region") or "目标区域"
    chart_type = evidence.get("visualization_type") or "visualization"
    image_grid = bool((evidence.get("grid_semantics") or {}).get("is_image_grid"))

    trend_text = "" if image_grid else describe_trend(temporal)
    hotspot_text = (
        f"热点连通域数量为 {spatial.get('hotspot_count')}，"
        f"高值连通域数量为 {spatial.get('connected_domains')}，"
        f"平均梯度强度为 {format_number(spatial.get('gradient_strength'))}。"
    )
    anomaly_text = ""
    if anomaly.get("count") is not None:
        anomaly_text = f" z-score 异常连通域数量为 {anomaly.get('count')}。"

    if image_grid:
        explanation = (
            f"{chart_type} 显示，该图像网格使用像素坐标，{variable} 平均值为 "
            f"{format_number(stats.get('mean'))}{unit_text}，最大值为 {format_number(stats.get('max'))}{unit_text}。"
            f"{hotspot_text}{anomaly_text} 这些结果描述的是像素亮度、边缘或纹理结构，不代表地理位置。"
        )
    else:
        explanation = (
            f"{chart_type} 显示，{region}{variable}有效区域均值为 "
            f"{format_number(stats.get('mean'))}{unit_text}，最大值为 {format_number(stats.get('max'))}{unit_text}。"
            f"{hotspot_text}{trend_text}{anomaly_text}"
        )
    if transformer.get("enabled"):
        model = transformer.get("model", {})
        saliency = transformer.get("saliency_summary", {})
        explanation += (
            f" Transformer 表征层将网格切分为 {model.get('token_count')} 个 patch token，"
            f"使用 {model.get('num_heads')} 个注意力头计算 patch 关系，"
            f"saliency p95 为 {format_number(saliency.get('p95'))}。"
            " 当前 saliency 融合了视觉显著性、注意力中心性和 token feature energy，用于减少纯注意力图的随机性。"
        )
    risks = evidence.get("uncertainty_inputs") or ["结论应限定在数据覆盖范围和有效区域内。"]

    return {
        "visualization_type": chart_type,
        "explanation": explanation,
        "uncertainty_risk": "；".join(str(item) for item in risks),
        "data_source": evidence.get("data_source"),
        "analysis_process": "；".join(evidence.get("analysis_process") or []),
        "calculation_parameters": "；".join(evidence.get("calculation_parameters") or []),
        "provenance": {
            "analysis_method": "Grid standardization -> Step 4 numeric/spatial/temporal analysis -> chart rendering -> Step 5 explanation",
            "tools_used": evidence.get("tools_used", []),
            "execution_log": evidence.get("execution_log_summary"),
            "artifacts": [item.get("path") for item in evidence.get("artifacts", [])],
            "reproducibility_notes": [
                "使用相同数据源、Step 3 plan 和 Step 4 参数可复现该报告。",
                "图像网格结论应解释为像素坐标和通道值分析。" if image_grid else "空间结论需结合有效区域与缺失率解释。",
            ],
        },
        "follow_up_questions": build_follow_up_questions(evidence),
        "llm_metadata": {"mode": "heuristic", "provider": None, "model": None, "fallback_used": True},
    }


def normalize_explanation(plan: Mapping[str, Any], *, fallback: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(fallback)
    for key in (
        "visualization_type",
        "explanation",
        "uncertainty_risk",
        "data_source",
        "analysis_process",
        "calculation_parameters",
        "follow_up_questions",
        "llm_metadata",
    ):
        value = plan.get(key)
        if value not in (None, "", [], {}):
            output[key] = value
    if isinstance(plan.get("provenance"), Mapping):
        provenance = dict(output.get("provenance", {}))
        provenance.update({k: v for k, v in plan["provenance"].items() if v not in (None, "", [], {})})
        output["provenance"] = provenance
    output.setdefault("follow_up_questions", build_follow_up_questions({}))
    output.setdefault("llm_metadata", fallback.get("llm_metadata", {}))
    return json_ready(output)


def write_step5_reports(
    output_dir: Path,
    explanation: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "step5_explanation.json"
    json_path.write_text(
        json.dumps(json_ready({"explanation": explanation, "evidence": evidence}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_path = output_dir / "step5_report.md"
    md_path.write_text(build_markdown_report(explanation, evidence), encoding="utf-8")

    html_path = output_dir / "step5_report.html"
    html_path.write_text(build_html_report(explanation, evidence), encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "html": str(html_path),
    }


def build_markdown_report(explanation: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    lines = [
        "# GridVis-LLM Step 5 Explanation and Provenance Report",
        "",
        "## Explanation",
        "",
        str(explanation.get("explanation", "")),
        "",
        "## Uncertainty and Risks",
        "",
        str(explanation.get("uncertainty_risk", "")),
        "",
        "## Provenance",
        "",
        f"- Visualization type: `{explanation.get('visualization_type')}`",
        f"- Data source: `{explanation.get('data_source')}`",
        f"- Analysis process: {explanation.get('analysis_process')}",
        f"- Calculation parameters: {explanation.get('calculation_parameters')}",
        "",
        "## Tools Used",
        "",
    ]
    for tool in explanation.get("provenance", {}).get("tools_used", []):
        lines.append(f"- {tool}")
    lines.extend(["", "## Artifacts", ""])
    for artifact in evidence.get("artifacts", []):
        lines.append(f"- `{artifact.get('type')}`: {artifact.get('path')}")
    if explanation.get("follow_up_questions"):
        lines.extend(["", "## Follow-up Questions", ""])
        lines.extend(f"- {item}" for item in explanation["follow_up_questions"])
    return "\n".join(lines) + "\n"


def build_html_report(explanation: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    figures = []
    for artifact in evidence.get("artifacts", []):
        path = Path(artifact.get("path", ""))
        if not path.exists():
            continue
        if path.suffix.lower() == ".png":
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            media = f'<img src="data:image/png;base64,{encoded}" alt="{escape(artifact.get("description", ""))}">'
        elif path.suffix.lower() == ".svg":
            media = path.read_text(encoding="utf-8")
        else:
            media = f'<a href="{escape(str(path))}">{escape(str(path))}</a>'
        figures.append(f"<section><h2>{escape(artifact.get('description', artifact.get('type', 'artifact')))}</h2>{media}</section>")

    tools = "".join(f"<li>{escape(str(tool))}</li>" for tool in explanation.get("provenance", {}).get("tools_used", []))
    questions = "".join(f"<li>{escape(str(item))}</li>" for item in explanation.get("follow_up_questions", []))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>GridVis-LLM Step 5 Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px auto; max-width: 1080px; color: #1f2933; line-height: 1.55; }}
    img, svg {{ max-width: 100%; height: auto; border: 1px solid #d6d9de; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
    section {{ margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>GridVis-LLM Step 5 Report</h1>
  <h2>解释文本</h2>
  <p>{escape(str(explanation.get('explanation', '')))}</p>
  <h2>不确定性与风险</h2>
  <p>{escape(str(explanation.get('uncertainty_risk', '')))}</p>
  <h2>溯源信息</h2>
  <ul>
    <li><strong>图表类型:</strong> {escape(str(explanation.get('visualization_type', '')))}</li>
    <li><strong>数据来源:</strong> <code>{escape(str(explanation.get('data_source', '')))}</code></li>
    <li><strong>分析过程:</strong> {escape(str(explanation.get('analysis_process', '')))}</li>
    <li><strong>计算参数:</strong> {escape(str(explanation.get('calculation_parameters', '')))}</li>
  </ul>
  <h2>工具链</h2>
  <ul>{tools}</ul>
  {''.join(figures)}
  <h2>后续问题</h2>
  <ul>{questions}</ul>
</body>
</html>
"""


def prepare_output_dir(output_dir: str | Path | None, bundle: Mapping[str, Any]) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        path = Path(bundle["base_dir"]) / "step5_explanation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_visualization_type(artifacts: Sequence[Mapping[str, Any]]) -> str:
    names = " ".join(str(item.get("name", "")) + " " + str(item.get("description", "")) for item in artifacts).lower()
    if any(str(item.get("type")) == "image" for item in artifacts) or "image" in names:
        return "image raster + intensity heatmap + edge/texture overlay"
    if "zscore" in names or "anomaly" in names:
        return "anomaly mask + z-score significance map"
    if "trajectory" in names:
        return "trajectory map + time series"
    if "scatter" in names:
        return "variable coupling scatter"
    return "heatmap + contour overlay"


def infer_tools_used(execution_log: Sequence[Mapping[str, Any]], artifacts: Sequence[Mapping[str, Any]]) -> list[str]:
    logged_tools = {str(item.get("tool")) for item in execution_log if item.get("tool")}
    tools = []
    if {"numeric", "spatial", "temporal"} & logged_tools:
        tools.append("numpy (array operations, statistics, trend inputs)")
    if "spatial" in logged_tools or "numeric" in logged_tools:
        tools.append("scipy.ndimage (connected components, nearest-neighbor fills, hotspot boundaries)")
    if any(str(item.get("type")) in {"map", "label_map"} for item in artifacts):
        tools.append("built-in PNG/SVG renderer (heatmap and map artifacts)")
    if any(str(item.get("type")) == "saliency_overlay" for item in artifacts):
        tools.append("built-in PNG/SVG renderer (saliency overlay artifacts)")
    if any(str(item.get("type")) == "image" for item in artifacts):
        tools.append("built-in PNG/SVG renderer (native image-grid preview)")
    if any(str(item.get("type")) in {"time_series", "trajectory", "scatter"} for item in artifacts):
        tools.append("built-in SVG renderer (series, trajectory, and scatter artifacts)")
    if "transformer" in logged_tools:
        tools.append("Grid Transformer encoder (patch tokenization, self-attention, saliency)")
    if not tools:
        tools.append("GridVis Step 4 execution pipeline")
    return unique_non_empty(tools)


def infer_process_steps(plan: Mapping[str, Any], execution_log: Sequence[Mapping[str, Any]]) -> list[str]:
    steps = [
        "Step 1: 将原始网格数据标准化为 GridTensor，并保留原始缺失掩膜。",
        "Step 4: 按任务图谱选择变量、区域和可视化策略。",
    ]
    for item in execution_log:
        tool = item.get("tool")
        message = item.get("message")
        if tool and message:
            steps.append(f"{tool}: {message}")
    chart_type = (plan.get("visualization_strategy") or {}).get("chart_type")
    if chart_type:
        steps.append(f"渲染策略: {chart_type}")
    return unique_non_empty(steps)


def infer_calculation_parameters(
    plan: Mapping[str, Any],
    numeric: Mapping[str, Any],
    spatial: Mapping[str, Any],
    temporal: Mapping[str, Any],
) -> list[str]:
    params = []
    stats = numeric.get("statistics", {})
    anomaly = numeric.get("anomaly", {})
    hotspots = spatial.get("hotspots", {})
    gradient = spatial.get("gradient", {})
    params.append(f"变量: {stats.get('variable')}")
    params.append(f"区域: {(plan.get('task_graph') or {}).get('region')}")
    params.append(f"图表类型: {(plan.get('visualization_strategy') or {}).get('chart_type')}")
    if anomaly.get("method"):
        params.append(f"异常检测: {anomaly.get('method')}，阈值 {anomaly.get('threshold')}")
    if hotspots.get("threshold") is not None:
        params.append(f"热点阈值: {format_number(hotspots.get('threshold'))}")
    if gradient.get("gradient_strength") is not None:
        params.append("梯度: numpy.gradient 计算空间变化强度")
    if temporal.get("trend", {}).get("trend"):
        params.append(f"趋势: {temporal['trend'].get('trend')}，斜率 {format_number(temporal['trend'].get('slope'))}")
    return unique_non_empty(params)


def infer_uncertainty(
    stats: Mapping[str, Any],
    trend: Mapping[str, Any],
    anomaly: Mapping[str, Any],
    *,
    image_grid: bool = False,
) -> list[str]:
    risks = []
    missing = stats.get("original_missing_ratio")
    if isinstance(missing, (int, float)) and missing > 0.3:
        risks.append(f"{stats.get('variable') or '目标变量'} 原始缺失率约 {missing:.1%}，边界和固定缺失区域解读需谨慎。")
    if image_grid:
        risks.append("该结果来自单帧图像网格，坐标为像素索引，不支持时序趋势、传播速度或地理区域解释。")
    elif trend.get("trend") == "single_time_step":
        risks.append("当前结果只有单个时间片，不能得出可靠趋势、周期或传播速度结论。")
    if anomaly.get("count") == 0:
        risks.append("z-score 方法未检测到显著异常，但这不排除局地小尺度异常。")
    return risks


def is_image_evidence(
    visualization_type: str | None,
    artifacts: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
) -> bool:
    text = str(visualization_type or "").lower()
    if "image" in text or "pixel" in text:
        return True
    if any(str(item.get("type")) == "image" for item in artifacts):
        return True
    return str(stats.get("variable", "")).lower() in {"intensity", "red", "green", "blue", "alpha"}


def summarize_artifacts(artifacts: Sequence[Mapping[str, Any]], base_dir: Path) -> list[dict[str, Any]]:
    result = []
    for item in artifacts:
        path = Path(item.get("path", ""))
        if path and not path.is_absolute():
            path = PROJECT_ROOT / path
        result.append(
            {
                "type": item.get("type"),
                "name": item.get("name"),
                "description": item.get("description"),
                "path": str(path) if path else item.get("path"),
                "exists": path.exists() if path else False,
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    if not result:
        for suffix in ("*.png", "*.svg", "*.html"):
            for path in base_dir.glob(suffix):
                result.append(
                    {
                        "type": path.suffix.lstrip("."),
                        "name": path.stem,
                        "description": path.name,
                        "path": str(path),
                        "exists": True,
                        "size_bytes": path.stat().st_size,
                    }
                )
    return result


def summarize_execution_log(execution_log: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "event_count": len(execution_log),
        "tools": unique_non_empty([item.get("tool") for item in execution_log if item.get("tool")]),
        "messages": [item.get("message") for item in execution_log[:12] if item.get("message")],
    }


def find_data_source_in_plan(plan: Mapping[str, Any]) -> str | None:
    source = plan.get("data_source") or plan.get("source")
    if source:
        return str(source)
    metadata = plan.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("data_source", "source", "path"):
            if metadata.get(key):
                return str(metadata[key])
    return None


def describe_trend(temporal: Mapping[str, Any]) -> str:
    trend = temporal.get("trend")
    if trend == "single_time_step":
        return " 当前仅有单个时间片，因此不支持可靠趋势判断。"
    if trend in {"increase", "decrease", "stable"}:
        return f" 时间序列趋势为 {trend}，斜率为 {format_number(temporal.get('slope'))}。"
    return ""


def build_follow_up_questions(evidence: Mapping[str, Any]) -> list[str]:
    task = evidence.get("task_graph", {}) if isinstance(evidence, Mapping) else {}
    variable = evidence.get("statistics", {}).get("variable") if isinstance(evidence, Mapping) else None
    if isinstance(evidence, Mapping) and (evidence.get("grid_semantics") or {}).get("is_image_grid"):
        return [
            "是否需要框选图像中的局部区域进行单独分析？",
            "是否需要分别比较 RGB 通道的亮度和纹理差异？",
            "是否需要生成边缘或异常像素掩膜？",
        ]
    questions = [
        "是否需要查看不同区域的对比结果？",
        "是否需要引入更多时间片来验证趋势或传播路径？",
    ]
    if variable:
        questions.append(f"是否需要分析其他变量对 {variable} 的影响？")
    if task.get("task_type") != "anomaly_detection":
        questions.append("是否需要进一步生成异常检测图和 z-score 显著性说明？")
    return questions[:3]


def load_json_if_exists(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_ready(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not (value == value and abs(value) != float("inf")):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not (number == number and abs(number) != float("inf")):
        return "NA"
    return f"{number:.3g}"


def unique_non_empty(values: Sequence[Any]) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Step 5 explanations and provenance reports from Step 4 outputs.")
    parser.add_argument("--step4-output", required=True, help="Step 4 output directory or analysis_results.json.")
    parser.add_argument("--data-source", help="Original data source path to include in provenance.")
    parser.add_argument("--output-dir", help="Directory for Step 5 report files. Defaults to <step4-output>/step5_explanation.")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic local explanation only.")
    parser.add_argument("--model", help="Override DashScope text model.")
    args = parser.parse_args()

    result = generate_result_explanation(
        args.step4_output,
        data_source=args.data_source,
        output_dir=args.output_dir,
        use_llm=not args.no_llm,
        model=args.model,
    )
    print(json.dumps(json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
