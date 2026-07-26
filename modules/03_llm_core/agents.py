from __future__ import annotations

import json
from typing import Any, Mapping

from llm_planner import DashScopeConfig, call_dashscope_chat, parse_llm_json


class GridVisAgent:
    def __init__(self, name: str, role: str, *, config: DashScopeConfig | None = None) -> None:
        self.name = name
        self.role = role
        self.config = config or DashScopeConfig.from_env()

    def run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self.config.api_key:
            raise RuntimeError("Missing DashScope API key.")

        messages = [
            {
                "role": "system",
                "content": (
                    f"你是 GridVis-LLM 的 {self.name}。{self.role}"
                    "只能输出合法 JSON，不要输出 Markdown。"
                    "必须基于传入 evidence，不要编造不存在的数值。"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        raw = call_dashscope_chat(messages, model=self.config.text_model, config=self.config)
        parsed = parse_llm_json(raw)
        parsed["agent"] = self.name
        return parsed


def compact_evidence(analysis: Mapping[str, Any], query: str) -> dict[str, Any]:
    selection = analysis.get("selection", {})
    stats = analysis.get("statistics", {})
    temporal = analysis.get("temporal", {})
    correlations = analysis.get("correlations", {}).get("daily", [])
    blocks = analysis.get("block_semantics", {}).get("top_blocks", [])
    hotspots = analysis.get("hotspots", [])
    return {
        "query": query,
        "selection": selection,
        "catalog": analysis.get("catalog", {}),
        "statistics": {
            "overall": stats.get("overall", {}),
            "latest": stats.get("latest", {}),
            "temporal_trend": stats.get("temporal_trend", {}),
            "quality": stats.get("quality", {}),
        },
        "top_hotspots": hotspots[:5],
        "top_correlations": correlations[:8],
        "top_semantic_blocks": blocks[:8],
        "temporal_samples": {
            "date_start": (temporal.get("dates") or [None])[0],
            "date_end": (temporal.get("dates") or [None])[-1],
            "selected_date": selection.get("selected_date"),
            "anomaly_days": temporal.get("anomaly_days", [])[:5],
        },
        "available_views": ["Heatmap", "Time Series", "Correlation View", "Semantic Blocks", "Trace & Provenance"],
    }


def run_llm_agents(query: str, analysis: Mapping[str, Any], *, config: DashScopeConfig | None = None) -> dict[str, Any]:
    config = config or DashScopeConfig.from_env()
    evidence = compact_evidence(analysis, query)

    task_agent = GridVisAgent(
        "TaskParsingAgent",
        "负责把自然语言任务转成 task_graph，包括任务类型、变量、区域、时间、分析目标。",
        config=config,
    )
    strategy_agent = GridVisAgent(
        "VisualizationStrategyAgent",
        "负责选择可视化视图、图层、交互、编码方式，并解释为什么这样组织界面。",
        config=config,
    )
    explanation_agent = GridVisAgent(
        "InsightExplanationAgent",
        "负责根据证据生成中文分析结论、风险提示、后续问题，语言应适合科研可视分析系统。",
        config=config,
    )

    task = task_agent.run(
        {
            "evidence": evidence,
            "required_schema": {
                "task_graph": {
                    "task_type": "distribution|trend_analysis|correlation_analysis|attribution|anomaly_detection|summary",
                    "region": "string",
                    "variables": ["string"],
                    "time_range": "string",
                    "analysis_goals": ["string"],
                    "constraints": ["string"],
                }
            },
        }
    )
    strategy = strategy_agent.run(
        {
            "evidence": evidence,
            "task_graph": task.get("task_graph", {}),
            "required_schema": {
                "visualization_strategy": {
                    "layout": "string",
                    "views": ["Heatmap", "Time Series", "Correlation View", "Semantic Blocks"],
                    "encodings": {"color": "string", "space": "string", "time": "string", "block": "string"},
                    "interaction": ["string"],
                    "reasoning": ["string"],
                }
            },
        }
    )
    explanation = explanation_agent.run(
        {
            "evidence": evidence,
            "task_graph": task.get("task_graph", {}),
            "visualization_strategy": strategy.get("visualization_strategy", {}),
            "required_schema": {
                "narrative": "中文结论，3-6句",
                "uncertainty_risks": ["string"],
                "follow_up_questions": ["string"],
            },
        }
    )

    return {
        "query": query,
        "task_graph": task.get("task_graph", {}),
        "visualization_strategy": strategy.get("visualization_strategy", {}),
        "narrative": explanation.get("narrative", ""),
        "uncertainty_risks": explanation.get("uncertainty_risks", []),
        "follow_up_questions": explanation.get("follow_up_questions", []),
        "agent_outputs": {
            "task": task,
            "strategy": strategy,
            "explanation": explanation,
        },
        "llm_metadata": {
            "mode": "dashscope_agents",
            "provider": "aliyun_dashscope",
            "model": config.text_model,
            "agents": ["TaskParsingAgent", "VisualizationStrategyAgent", "InsightExplanationAgent"],
            "fallback_used": False,
        },
    }
