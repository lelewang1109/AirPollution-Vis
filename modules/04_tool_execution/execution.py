from __future__ import annotations

import argparse
import base64
import json
import math
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULES_DIR = PROJECT_ROOT / "modules"
DATA_ADAPTER_DIR = MODULES_DIR / "01_data_adapter"
LLM_CORE_DIR = MODULES_DIR / "03_llm_core"
GRID_REPRESENTATION_DIR = MODULES_DIR / "02_grid_representation"
for path in (DATA_ADAPTER_DIR, LLM_CORE_DIR, GRID_REPRESENTATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adapter import GridTensor, load_grid_data  # noqa: E402
from transformer_grid_encoder import encode_grid_transformer  # noqa: E402


REGION_BOUNDS = {
    "中国全境": (73.0, 135.5, 18.0, 54.5),
    "数据覆盖区域": None,
    "华北地区": (105.0, 125.0, 34.0, 43.5),
    "华北区域": (105.0, 125.0, 34.0, 43.5),
    "京津冀": (113.0, 120.5, 36.0, 42.5),
    "东北地区": (118.0, 135.5, 38.0, 54.5),
    "华东地区": (114.0, 123.5, 24.0, 38.5),
    "华南地区": (105.0, 122.5, 18.0, 26.5),
    "西南地区": (97.0, 112.5, 21.0, 35.5),
    "西北地区": (73.0, 112.5, 32.0, 50.5),
    "长三角": (116.0, 123.5, 28.0, 34.5),
    "珠三角": (111.0, 116.0, 21.0, 24.5),
}

SEQUENTIAL_COLORS = [
    (247, 252, 245),
    (229, 245, 224),
    (199, 233, 192),
    (161, 217, 155),
    (116, 196, 118),
    (65, 171, 93),
    (35, 139, 69),
    (0, 109, 44),
    (0, 68, 27),
]

DIVERGING_COLORS = [
    (49, 54, 149),
    (69, 117, 180),
    (116, 173, 209),
    (171, 217, 233),
    (224, 243, 248),
    (255, 255, 191),
    (254, 224, 144),
    (253, 174, 97),
    (244, 109, 67),
    (215, 48, 39),
    (165, 0, 38),
]


@dataclass
class ExecutionContext:
    plan: dict[str, Any]
    grid_tensor: GridTensor
    output_dir: Path
    output_format: str = "png"
    variable: str | None = None
    region: str = "数据覆盖区域"
    logs: list[dict[str, Any]] = field(default_factory=list)

    def log(self, tool: str, message: str, **details: Any) -> None:
        self.logs.append(
            {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "tool": tool,
                "message": message,
                "details": json_ready(details),
            }
        )


def execute_plan(
    plan: Mapping[str, Any],
    source: str | Path | GridTensor | Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    output_format: str = "png",
    variables: Sequence[str] | None = None,
    fill_missing: bool = True,
) -> dict[str, Any]:
    plan_dict = normalize_plan(plan)
    grid_tensor = coerce_grid_tensor(source, variables=variables, fill_missing=fill_missing)
    output_path = prepare_output_dir(output_dir)

    selected_variable = choose_variable(plan_dict, grid_tensor, variables)
    region = plan_dict.get("task_graph", {}).get("region") or "数据覆盖区域"
    ctx = ExecutionContext(
        plan=plan_dict,
        grid_tensor=grid_tensor,
        output_dir=output_path,
        output_format=output_format.lower(),
        variable=selected_variable,
        region=region,
    )

    ctx.log("executor", "Started Step 4 execution", task_type=get_task_type(plan_dict), variable=selected_variable)
    prepared = prepare_analysis_arrays(ctx)
    numeric = run_numeric_analysis(ctx, prepared)
    spatial = run_spatial_analysis(ctx, prepared)
    temporal = run_temporal_analysis(ctx, prepared)
    transformer = run_transformer_analysis(ctx, prepared)
    artifacts = render_visual_outputs(ctx, prepared, numeric, spatial, temporal, transformer)
    explanation = build_explanation(ctx, numeric, spatial, temporal, transformer, artifacts)
    report_files = write_reports(ctx, numeric, spatial, temporal, transformer, artifacts, explanation)
    log_file = write_execution_log(ctx)

    return {
        "artifacts": artifacts,
        "analysis_results": {
            "numeric": strip_large_arrays(numeric),
            "spatial": strip_large_arrays(spatial),
            "temporal": strip_large_arrays(temporal),
            "transformer": strip_large_arrays(transformer),
        },
        "explanation": explanation,
        "report_files": report_files,
        "execution_log": str(log_file),
        "output_dir": str(output_path),
    }


def coerce_grid_tensor(
    source: str | Path | GridTensor | Mapping[str, Any],
    *,
    variables: Sequence[str] | None,
    fill_missing: bool,
) -> GridTensor:
    if isinstance(source, GridTensor):
        return source
    if isinstance(source, Mapping) and isinstance(source.get("grid_tensor"), GridTensor):
        return source["grid_tensor"]
    loaded = load_grid_data(source, variables=variables, fill_missing=fill_missing)
    return loaded["grid_tensor"]


def normalize_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if "task_graph" in plan or "visualization_strategy" in plan:
        normalized = dict(plan)
    else:
        normalized = {
            "task_graph": {
                "task_type": plan.get("task_type", "distribution"),
                "region": plan.get("region", "数据覆盖区域"),
                "variables": [plan["variable"]] if plan.get("variable") else [],
                "time_range": plan.get("time_range", "数据时间范围"),
            },
            "visualization_strategy": plan.get("visualization_strategy", {}),
            "uncertainty_risks": plan.get("uncertainty_risks", []),
        }

    normalized.setdefault("task_graph", {})
    normalized.setdefault("visualization_strategy", {})
    normalized.setdefault("analysis_plan", {})
    normalized.setdefault("uncertainty_risks", [])
    normalized.setdefault("follow_up_questions", [])
    normalized["task_graph"].setdefault("task_type", "distribution")
    normalized["task_graph"].setdefault("region", "数据覆盖区域")
    normalized["task_graph"].setdefault("variables", [])
    normalized["task_graph"].setdefault("time_range", "数据时间范围")
    normalized["visualization_strategy"].setdefault("chart_type", "heatmap + contour overlay")
    normalized["visualization_strategy"].setdefault("layout", "single map with summary cards")
    normalized["visualization_strategy"].setdefault("color_encoding", "sequential color scale")
    normalized["visualization_strategy"].setdefault("focus_region", normalized["task_graph"]["region"])
    return normalized


def choose_variable(
    plan: Mapping[str, Any],
    grid_tensor: GridTensor,
    variables: Sequence[str] | None,
) -> str:
    candidates = list(variables or [])
    candidates.extend(plan.get("task_graph", {}).get("variables") or [])
    for candidate in candidates:
        if candidate in grid_tensor.variables:
            return str(candidate)
        match = match_variable(str(candidate), grid_tensor.variables)
        if match:
            return match
    return grid_tensor.variables[0]


def match_variable(candidate: str, variables: Sequence[str]) -> str | None:
    lower = candidate.lower()
    aliases = {
        "pm25": "pm2.5",
        "pm 2.5": "pm2.5",
        "temperature": "temp",
        "humidity": "rhum",
    }
    lower = aliases.get(lower, lower)
    for variable in variables:
        if variable.lower() == lower:
            return variable
    for variable in variables:
        if lower in variable.lower() or variable.lower() in lower:
            return variable
    return None


def prepare_analysis_arrays(ctx: ExecutionContext) -> dict[str, Any]:
    grid_tensor = ctx.grid_tensor
    variable_index = grid_tensor.variables.index(ctx.variable)
    data = np.array(grid_tensor.data[:, variable_index], dtype=float, copy=True)
    original_missing = np.array(grid_tensor.mask.get("original_missing")[:, variable_index], dtype=bool, copy=True)
    valid_region = np.array(grid_tensor.mask.get("valid_region", np.ones(data.shape[-2:], dtype=bool)), dtype=bool)

    lon = np.asarray(grid_tensor.space["coordinates"]["lon"], dtype=float)
    lat = np.asarray(grid_tensor.space["coordinates"]["lat"], dtype=float)
    region_mask = build_region_mask(ctx.region, lon, lat)
    combined_region = valid_region & region_mask
    data = np.where(combined_region[np.newaxis, :, :], data, np.nan)

    ctx.log(
        "data_preparation",
        "Prepared variable and region subset",
        variable=ctx.variable,
        region=ctx.region,
        time_steps=data.shape[0],
        valid_cells=int(np.count_nonzero(combined_region)),
    )

    return {
        "data": data,
        "original_missing": original_missing,
        "valid_region": valid_region,
        "region_mask": region_mask,
        "combined_region": combined_region,
        "lon": lon,
        "lat": lat,
        "time": grid_tensor.time,
        "variable": ctx.variable,
        "variable_unit": grid_tensor.variable_units.get(ctx.variable),
    }


def build_region_mask(region: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    bounds = REGION_BOUNDS.get(region)
    if bounds is None:
        return np.ones((len(lat), len(lon)), dtype=bool)
    lon_min, lon_max, lat_min, lat_max = bounds
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    return lat_mask[:, np.newaxis] & lon_mask[np.newaxis, :]


def run_numeric_analysis(ctx: ExecutionContext, prepared: Mapping[str, Any]) -> dict[str, Any]:
    data = prepared["data"]
    original_missing = prepared["original_missing"]
    values = data[np.isfinite(data)]
    stats = {
        "variable": prepared["variable"],
        "unit": prepared["variable_unit"],
        "min": safe_float(np.nanmin(values)) if values.size else None,
        "max": safe_float(np.nanmax(values)) if values.size else None,
        "mean": safe_float(np.nanmean(values)) if values.size else None,
        "std": safe_float(np.nanstd(values)) if values.size else None,
        "median": safe_float(np.nanmedian(values)) if values.size else None,
        "p05": safe_float(np.nanpercentile(values, 5)) if values.size else None,
        "p95": safe_float(np.nanpercentile(values, 95)) if values.size else None,
        "valid_count": int(values.size),
        "original_missing_ratio": safe_ratio(np.count_nonzero(original_missing), original_missing.size),
    }
    anomaly = detect_anomalies(data)
    trend = compute_linear_trend(data)

    ctx.log("numeric", "Computed statistics, anomaly scores, and trend metrics", stats=stats)
    return {
        "statistics": stats,
        "anomaly": anomaly,
        "trend": trend,
    }


def detect_anomalies(data: np.ndarray) -> dict[str, Any]:
    task_map = data[-1] if data.shape[0] else np.empty((0, 0))
    finite = np.isfinite(task_map)
    if not finite.any():
        return {"method": "z_score", "threshold": 2.0, "count": 0, "locations": [], "z_map": task_map}
    mean = float(np.nanmean(task_map))
    std = float(np.nanstd(task_map))
    if std == 0:
        z_map = np.zeros_like(task_map)
    else:
        z_map = (task_map - mean) / std
    mask = finite & (np.abs(z_map) >= 2.0)
    labeled, count = ndimage.label(mask)
    return {
        "method": "z_score",
        "threshold": 2.0,
        "count": int(count),
        "locations": component_summaries(labeled, task_map, limit=10),
        "z_map": z_map,
    }


def compute_linear_trend(data: np.ndarray) -> dict[str, Any]:
    series = spatial_mean_series(data)
    finite = np.isfinite(series)
    if np.count_nonzero(finite) < 2:
        return {"trend": "single_time_step", "slope": 0.0, "series": json_ready(series)}
    x = np.arange(series.size, dtype=float)[finite]
    y = series[finite]
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    trend = "increase" if slope > 0 else "decrease" if slope < 0 else "stable"
    return {
        "trend": trend,
        "slope": safe_float(slope),
        "intercept": safe_float(intercept),
        "r2": safe_float(r2),
        "series": json_ready(series),
    }


def run_spatial_analysis(ctx: ExecutionContext, prepared: Mapping[str, Any]) -> dict[str, Any]:
    data = prepared["data"]
    mean_map = nanmean_time(data)
    hotspot = detect_hotspots(mean_map)
    gradient = compute_gradient_features(mean_map)
    ctx.log(
        "spatial",
        "Detected hotspots, connected domains, and gradients",
        hotspot_count=hotspot["hotspot_count"],
        connected_domains=hotspot["connected_domains"],
    )
    return {
        "mean_map": mean_map,
        "hotspots": hotspot,
        "gradient": gradient,
    }


def detect_hotspots(values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    if not finite.any():
        empty = np.zeros_like(values, dtype=bool)
        return {
            "threshold": None,
            "hotspot_count": 0,
            "connected_domains": 0,
            "hotspot_mask": empty,
            "labeled_hotspots": np.zeros_like(values, dtype=int),
            "components": [],
        }
    threshold = float(np.nanquantile(values[finite], 0.9))
    hotspot_mask = finite & (values >= threshold)
    labeled, hotspot_count = ndimage.label(hotspot_mask)
    high_mask = finite & (values >= np.nanmean(values[finite]) + np.nanstd(values[finite]))
    _, connected_domains = ndimage.label(high_mask)
    return {
        "threshold": safe_float(threshold),
        "hotspot_count": int(hotspot_count),
        "connected_domains": int(connected_domains),
        "hotspot_mask": hotspot_mask,
        "labeled_hotspots": labeled,
        "components": component_summaries(labeled, values, limit=10),
    }


def component_summaries(labels: np.ndarray, values: np.ndarray, *, limit: int) -> list[dict[str, Any]]:
    summaries = []
    for label_id in range(1, int(np.nanmax(labels)) + 1 if labels.size else 1):
        mask = labels == label_id
        if not np.any(mask):
            continue
        peak_index = np.nanargmax(np.where(mask, values, np.nan))
        peak_y, peak_x = np.unravel_index(peak_index, values.shape)
        yy, xx = np.where(mask)
        summaries.append(
            {
                "component": int(label_id),
                "cell_count": int(mask.sum()),
                "centroid_index": [safe_float(np.mean(yy)), safe_float(np.mean(xx))],
                "peak_index": [int(peak_y), int(peak_x)],
                "peak_value": safe_float(values[peak_y, peak_x]),
            }
        )
    summaries.sort(key=lambda item: item["cell_count"], reverse=True)
    return summaries[:limit]


def compute_gradient_features(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2 or not np.isfinite(values).any():
        return {
            "gradient_strength": None,
            "gradient_p95": None,
            "anisotropy": None,
            "boundary_change": None,
            "gradient_map": np.zeros_like(values, dtype=float),
        }
    filled = fill_nan_nearest(values)
    grad_y, grad_x = np.gradient(filled)
    gradient_map = np.hypot(grad_x, grad_y)
    finite = np.isfinite(values)
    x_energy = float(np.nanmean(grad_x[finite] ** 2)) if np.any(finite) else 0.0
    y_energy = float(np.nanmean(grad_y[finite] ** 2)) if np.any(finite) else 0.0
    anisotropy = abs(x_energy - y_energy) / (x_energy + y_energy) if (x_energy + y_energy) else 0.0
    boundary = gradient_map >= np.nanquantile(gradient_map[finite], 0.9) if np.any(finite) else np.zeros_like(finite)
    return {
        "gradient_strength": safe_float(np.nanmean(gradient_map[finite])) if np.any(finite) else None,
        "gradient_p95": safe_float(np.nanpercentile(gradient_map[finite], 95)) if np.any(finite) else None,
        "anisotropy": safe_float(anisotropy),
        "boundary_change": safe_float(np.nanmean(gradient_map[boundary])) if np.any(boundary) else None,
        "gradient_map": gradient_map,
    }


def run_temporal_analysis(ctx: ExecutionContext, prepared: Mapping[str, Any]) -> dict[str, Any]:
    data = prepared["data"]
    series = spatial_mean_series(data)
    trend = compute_linear_trend(data)
    change_points = detect_change_points(series, prepared["time"])
    ctx.log("temporal", "Computed temporal series, trend, and change points", trend=trend["trend"])
    return {
        "time": prepared["time"],
        "series": json_ready(series),
        "trend": trend,
        "change_points": change_points,
    }


def run_transformer_analysis(ctx: ExecutionContext, prepared: Mapping[str, Any]) -> dict[str, Any]:
    patch_shape = (64, 64) if is_image_grid(ctx.grid_tensor) else (32, 32)
    result = encode_grid_transformer(
        ctx.grid_tensor,
        patch_shape=patch_shape,
        model_dim=32,
        num_heads=4,
        include_saliency_map=True,
    )
    model = result.get("model", {})
    ctx.log(
        "transformer",
        "Encoded grid patches with lightweight Transformer self-attention",
        token_count=model.get("token_count"),
        model=model.get("name"),
    )
    return result


def detect_change_points(series: np.ndarray, time_values: Sequence[str | None]) -> list[dict[str, Any]]:
    finite = np.isfinite(series)
    valid = series[finite]
    valid_time = [time_values[idx] for idx, ok in enumerate(finite) if ok]
    if valid.size < 3:
        return []
    diffs = np.diff(valid)
    scale = float(np.nanstd(diffs))
    if scale == 0 or not np.isfinite(scale):
        return []
    threshold = max(2.0 * scale, float(np.nanpercentile(np.abs(diffs), 90)))
    indices = np.where(np.abs(diffs) >= threshold)[0] + 1
    return [
        {
            "time": valid_time[idx] if idx < len(valid_time) else str(idx),
            "delta": safe_float(diffs[idx - 1]),
            "index": int(idx),
        }
        for idx in indices[:10]
    ]


def render_visual_outputs(
    ctx: ExecutionContext,
    prepared: Mapping[str, Any],
    numeric: Mapping[str, Any],
    spatial: Mapping[str, Any],
    temporal: Mapping[str, Any],
    transformer: Mapping[str, Any],
) -> list[dict[str, Any]]:
    task_type = get_task_type(ctx.plan)
    chart_type = ctx.plan["visualization_strategy"].get("chart_type", "")
    artifacts: list[dict[str, Any]] = []
    if is_image_grid(ctx.grid_tensor):
        artifacts.append(render_rgb_image_artifact(ctx, "image_rgb_preview"))
        if transformer.get("saliency_map") is not None:
            artifacts.extend(render_transformer_saliency_artifacts(ctx, transformer, "transformer_saliency"))

    if task_type == "anomaly_detection" or "z-score" in chart_type or "anomaly" in chart_type:
        artifacts.append(render_map_artifact(ctx, numeric["anomaly"]["z_map"], "anomaly_zscore", diverging=True, title="Anomaly z-score map"))
    elif task_type in {"trend_analysis", "source_tracing"} and len(prepared["time"]) > 1:
        artifacts.append(render_map_artifact(ctx, spatial["mean_map"], "mean_heatmap", diverging=False, title=f"{ctx.variable} mean heatmap"))
        artifacts.append(render_series_artifact(ctx, temporal["series"], prepared["time"], "time_series", title=f"{ctx.variable} time series"))
        trajectory = build_hotspot_trajectory(prepared["data"])
        artifacts.append(render_trajectory_artifact(ctx, trajectory, prepared, "hotspot_trajectory"))
    elif task_type == "comparison" and prepared["data"].shape[0] > 1:
        diff = prepared["data"][-1] - prepared["data"][0]
        artifacts.append(render_map_artifact(ctx, diff, "difference_map", diverging=True, title=f"{ctx.variable} difference map"))
    else:
        artifacts.append(
            render_map_artifact(
                ctx,
                spatial["mean_map"],
                "heatmap_contour",
                diverging=False,
                title=f"{ctx.variable} heatmap with contour overlay",
                contour_mask=spatial["hotspots"]["hotspot_mask"],
            )
        )

    if task_type in {"clustering", "source_tracing"}:
        artifacts.append(render_label_artifact(ctx, spatial["hotspots"]["labeled_hotspots"], "hotspot_domains"))
    if task_type in {"attribution", "correlation_analysis"} and len(ctx.grid_tensor.variables) > 1:
        pair = choose_secondary_variable(ctx)
        if pair:
            artifacts.append(render_scatter_artifact(ctx, prepared, pair, "variable_coupling_scatter"))

    html_file = write_interactive_html(ctx, artifacts)
    artifacts.append({"type": "interactive_html", "path": str(html_file), "description": "HTML page containing generated charts"})
    ctx.log("visualization", "Rendered visual artifacts", artifact_count=len(artifacts))
    return artifacts


def render_map_artifact(
    ctx: ExecutionContext,
    values: np.ndarray,
    name: str,
    *,
    diverging: bool,
    title: str,
    contour_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    image = map_to_rgb(values, diverging=diverging)
    if contour_mask is not None:
        image = overlay_mask_boundary(image, contour_mask, color=(30, 30, 30))
    png_path = ctx.output_dir / f"{name}.png"
    svg_path = ctx.output_dir / f"{name}.svg"
    write_png(png_path, image)
    write_svg_heatmap(svg_path, image, title=title)
    return {
        "type": "map",
        "name": name,
        "path": str(png_path if ctx.output_format in {"png", "html"} else svg_path),
        "png": str(png_path),
        "svg": str(svg_path),
        "description": title,
    }


def render_rgb_image_artifact(ctx: ExecutionContext, name: str) -> dict[str, Any]:
    image = build_rgb_image(ctx.grid_tensor)
    png_path = ctx.output_dir / f"{name}.png"
    svg_path = ctx.output_dir / f"{name}.svg"
    write_png(png_path, image)
    write_svg_heatmap(svg_path, image, title="Original image grid RGB preview")
    return {
        "type": "image",
        "name": name,
        "path": str(png_path if ctx.output_format in {"png", "html"} else svg_path),
        "png": str(png_path),
        "svg": str(svg_path),
        "description": "Original image grid RGB preview",
    }


def render_transformer_saliency_artifacts(
    ctx: ExecutionContext,
    transformer: Mapping[str, Any],
    name: str,
) -> list[dict[str, Any]]:
    saliency_map = np.asarray(transformer.get("saliency_map"), dtype=float)
    heatmap = render_map_artifact(
        ctx,
        saliency_map,
        name,
        diverging=False,
        title="Adaptive Transformer + visual saliency map",
    )
    overlay = saliency_overlay_artifact(ctx, saliency_map, f"{name}_overlay")
    return [heatmap, overlay]


def saliency_overlay_artifact(ctx: ExecutionContext, saliency_map: np.ndarray, name: str) -> dict[str, Any]:
    rgb = build_rgb_image(ctx.grid_tensor)
    saliency = normalize_map(saliency_map)
    heat = saliency_to_heat_rgb(saliency)
    alpha = np.clip(saliency[..., np.newaxis] * 0.65, 0, 0.65)
    overlay = np.round(rgb * (1 - alpha) + heat * alpha).astype(np.uint8)
    png_path = ctx.output_dir / f"{name}.png"
    svg_path = ctx.output_dir / f"{name}.svg"
    write_png(png_path, overlay)
    write_svg_heatmap(svg_path, overlay, title="Transformer saliency overlay on original image")
    return {
        "type": "saliency_overlay",
        "name": name,
        "path": str(png_path if ctx.output_format in {"png", "html"} else svg_path),
        "png": str(png_path),
        "svg": str(svg_path),
        "description": "Transformer saliency overlay on original image",
    }


def saliency_to_heat_rgb(saliency: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(saliency, dtype=float), 0, 1)
    colors = [
        (0, 0, 0),
        (0, 80, 180),
        (0, 180, 220),
        (255, 210, 0),
        (255, 60, 40),
    ]
    flat = interpolate_colors(values.reshape(-1), colors)
    return flat.reshape((*values.shape, 3))


def build_rgb_image(grid_tensor: GridTensor) -> np.ndarray:
    names = [name.lower() for name in grid_tensor.variables]
    if all(name in names for name in ("red", "green", "blue")):
        channels = []
        for name in ("red", "green", "blue"):
            idx = names.index(name)
            channels.append(scale_channel(nanmean_time(grid_tensor.data[:, idx])))
        return np.dstack(channels).astype(np.uint8)

    if "intensity" in names:
        idx = names.index("intensity")
        gray = scale_channel(nanmean_time(grid_tensor.data[:, idx]))
    else:
        gray = scale_channel(nanmean_time(np.nanmean(grid_tensor.data, axis=1)))
    return np.dstack([gray, gray, gray]).astype(np.uint8)


def scale_channel(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if not finite.any():
        return out
    vmin = float(np.nanmin(arr[finite]))
    vmax = float(np.nanmax(arr[finite]))
    if vmin >= 0 and vmax <= 255:
        scaled = np.clip(arr, 0, 255)
    elif vmax == vmin:
        scaled = np.full(arr.shape, 127.0)
    else:
        scaled = (arr - vmin) / (vmax - vmin) * 255.0
    out[finite] = np.clip(np.round(scaled[finite]), 0, 255).astype(np.uint8)
    return out


def normalize_map(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=float)
    if not finite.any():
        return out
    vmin = float(np.nanmin(arr[finite]))
    vmax = float(np.nanmax(arr[finite]))
    if vmax == vmin:
        return out
    out[finite] = (arr[finite] - vmin) / (vmax - vmin)
    return out


def is_image_grid(grid_tensor: GridTensor) -> bool:
    metadata = grid_tensor.metadata or {}
    dataset_attrs = metadata.get("dataset_attrs") or {}
    if metadata.get("grid_type") == "image" or dataset_attrs.get("grid_type") == "image":
        return True
    image_names = {"intensity", "red", "green", "blue", "alpha", "gray", "grayscale"}
    return bool(set(name.lower() for name in grid_tensor.variables) & image_names)


def render_label_artifact(ctx: ExecutionContext, labels: np.ndarray, name: str) -> dict[str, Any]:
    image = label_to_rgb(labels)
    png_path = ctx.output_dir / f"{name}.png"
    svg_path = ctx.output_dir / f"{name}.svg"
    write_png(png_path, image)
    write_svg_heatmap(svg_path, image, title="Hotspot connected domains")
    return {
        "type": "label_map",
        "name": name,
        "path": str(png_path if ctx.output_format in {"png", "html"} else svg_path),
        "png": str(png_path),
        "svg": str(svg_path),
        "description": "Connected hotspot domain labels",
    }


def render_series_artifact(
    ctx: ExecutionContext,
    series: Sequence[float],
    time_values: Sequence[str | None],
    name: str,
    *,
    title: str,
) -> dict[str, Any]:
    svg = build_line_svg(series, time_values, title=title)
    svg_path = ctx.output_dir / f"{name}.svg"
    svg_path.write_text(svg, encoding="utf-8")
    return {
        "type": "time_series",
        "name": name,
        "path": str(svg_path),
        "svg": str(svg_path),
        "description": title,
    }


def build_hotspot_trajectory(data: np.ndarray) -> list[dict[str, Any]]:
    trajectory = []
    for tidx in range(data.shape[0]):
        values = data[tidx]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        threshold = float(np.nanquantile(values[finite], 0.9))
        mask = finite & (values >= threshold)
        if not np.any(mask):
            continue
        yy, xx = np.where(mask)
        trajectory.append(
            {
                "time_index": tidx,
                "centroid_index": [safe_float(np.mean(yy)), safe_float(np.mean(xx))],
                "cell_count": int(mask.sum()),
                "threshold": safe_float(threshold),
            }
        )
    return trajectory


def render_trajectory_artifact(
    ctx: ExecutionContext,
    trajectory: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    svg = build_trajectory_svg(trajectory, prepared["data"].shape[-2:], title="Hotspot centroid trajectory")
    svg_path = ctx.output_dir / f"{name}.svg"
    svg_path.write_text(svg, encoding="utf-8")
    return {
        "type": "trajectory",
        "name": name,
        "path": str(svg_path),
        "svg": str(svg_path),
        "description": "Hotspot centroid trajectory",
        "trajectory": json_ready(trajectory),
    }


def choose_secondary_variable(ctx: ExecutionContext) -> str | None:
    for variable in ctx.grid_tensor.variables:
        if variable != ctx.variable:
            return variable
    return None


def render_scatter_artifact(
    ctx: ExecutionContext,
    prepared: Mapping[str, Any],
    secondary_variable: str,
    name: str,
) -> dict[str, Any]:
    primary = prepared["data"].reshape(-1)
    secondary_idx = ctx.grid_tensor.variables.index(secondary_variable)
    secondary = ctx.grid_tensor.data[:, secondary_idx].reshape(-1)
    valid = np.isfinite(primary) & np.isfinite(secondary)
    primary = primary[valid]
    secondary = secondary[valid]
    if primary.size > 3000:
        step = max(1, primary.size // 3000)
        primary = primary[::step]
        secondary = secondary[::step]
    svg = build_scatter_svg(primary, secondary, xlabel=ctx.variable, ylabel=secondary_variable)
    svg_path = ctx.output_dir / f"{name}.svg"
    svg_path.write_text(svg, encoding="utf-8")
    return {
        "type": "scatter",
        "name": name,
        "path": str(svg_path),
        "svg": str(svg_path),
        "description": f"{ctx.variable} and {secondary_variable} coupling scatter",
    }


def write_interactive_html(ctx: ExecutionContext, artifacts: Sequence[Mapping[str, Any]]) -> Path:
    sections = []
    for artifact in artifacts:
        path = Path(artifact.get("png") or artifact.get("svg") or artifact["path"])
        if path.suffix == ".png":
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            media = f'<img src="data:image/png;base64,{encoded}" alt="{escape(artifact.get("description", ""))}">'
        else:
            media = path.read_text(encoding="utf-8")
        sections.append(
            f"<section><h2>{escape(artifact.get('description', artifact.get('name', 'chart')))}</h2>{media}</section>"
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>GridVis Execution Result</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }}
    section {{ margin-bottom: 28px; }}
    img, svg {{ max-width: 100%; height: auto; border: 1px solid #d6d9de; }}
    h1, h2 {{ font-weight: 650; }}
  </style>
</head>
<body>
  <h1>GridVis Tool Execution</h1>
  <p>Task: {escape(get_task_type(ctx.plan))}; Variable: {escape(str(ctx.variable))}; Region: {escape(ctx.region)}</p>
  {''.join(sections)}
</body>
</html>
"""
    path = ctx.output_dir / "visualization.html"
    path.write_text(html, encoding="utf-8")
    return path


def build_explanation(
    ctx: ExecutionContext,
    numeric: Mapping[str, Any],
    spatial: Mapping[str, Any],
    temporal: Mapping[str, Any],
    transformer: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stats = numeric["statistics"]
    hotspot = spatial["hotspots"]
    gradient = spatial["gradient"]
    trend = temporal["trend"]
    unit_text = f" {stats.get('unit')}" if stats.get("unit") else ""
    risks = list(ctx.plan.get("uncertainty_risks") or [])
    if stats.get("original_missing_ratio", 0) > 0.3:
        risks.append(f"{ctx.variable} 原始缺失率约 {stats['original_missing_ratio']:.1%}，图表应结合有效区域掩膜解释。")
    if trend.get("trend") == "single_time_step":
        risks.append("当前只有单个时间片，趋势和传播路径不能作为确定结论。")

    summary = (
        f"{ctx.region}{ctx.variable}有效区域均值为 {format_number(stats.get('mean'))}"
        f"{unit_text}，最大值为 {format_number(stats.get('max'))}{unit_text}。"
        f"热点连通域数量为 {hotspot.get('hotspot_count')}，平均梯度强度为 {format_number(gradient.get('gradient_strength'))}。"
    )
    if trend.get("trend") != "single_time_step":
        summary += f" 时间序列趋势为 {trend.get('trend')}，斜率为 {format_number(trend.get('slope'))}。"
    if transformer.get("enabled"):
        model = transformer.get("model", {})
        saliency = transformer.get("saliency_summary", {})
        summary += (
            f" Grid Transformer 使用 {model.get('token_count')} 个 patch token 和 "
            f"{model.get('num_heads')} 个注意力头，patch saliency p95 为 {format_number(saliency.get('p95'))}。"
        )

    return {
        "summary": summary,
        "key_findings": [
            f"均值: {format_number(stats.get('mean'))}{unit_text}",
            f"最大值: {format_number(stats.get('max'))}{unit_text}",
            f"热点区域数量: {hotspot.get('hotspot_count')}",
            f"连通高值域数量: {hotspot.get('connected_domains')}",
            f"梯度强度: {format_number(gradient.get('gradient_strength'))}",
            f"时间趋势: {trend.get('trend')}",
            f"Transformer token 数: {transformer.get('model', {}).get('token_count')}",
            f"Transformer saliency p95: {format_number(transformer.get('saliency_summary', {}).get('p95'))}",
        ],
        "risks": unique(risks),
        "artifact_count": len(artifacts),
    }


def write_reports(
    ctx: ExecutionContext,
    numeric: Mapping[str, Any],
    spatial: Mapping[str, Any],
    temporal: Mapping[str, Any],
    transformer: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    explanation: Mapping[str, Any],
) -> dict[str, str]:
    analysis_json = ctx.output_dir / "analysis_results.json"
    analysis_json.write_text(
        json.dumps(
            json_ready(
                strip_large_arrays(
                    {
                        "numeric": numeric,
                        "spatial": spatial,
                        "temporal": temporal,
                        "transformer": transformer,
                        "artifacts": artifacts,
                        "explanation": explanation,
                        "plan": ctx.plan,
                    }
                )
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_md = ctx.output_dir / "report.md"
    report_md.write_text(build_markdown_report(ctx, numeric, spatial, temporal, transformer, artifacts, explanation), encoding="utf-8")

    report_html = ctx.output_dir / "report.html"
    report_html.write_text(build_html_report(ctx, artifacts, explanation), encoding="utf-8")
    ctx.log("report", "Wrote analysis reports", report=str(report_md), json=str(analysis_json))
    return {
        "analysis_json": str(analysis_json),
        "markdown": str(report_md),
        "html": str(report_html),
    }


def build_markdown_report(
    ctx: ExecutionContext,
    numeric: Mapping[str, Any],
    spatial: Mapping[str, Any],
    temporal: Mapping[str, Any],
    transformer: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    explanation: Mapping[str, Any],
) -> str:
    stats = numeric["statistics"]
    hotspot = spatial["hotspots"]
    gradient = spatial["gradient"]
    lines = [
        "# GridVis Tool Execution Report",
        "",
        f"- Task type: `{get_task_type(ctx.plan)}`",
        f"- Region: `{ctx.region}`",
        f"- Variable: `{ctx.variable}`",
        f"- Chart strategy: `{ctx.plan['visualization_strategy'].get('chart_type')}`",
        "",
        "## Summary",
        "",
        explanation["summary"],
        "",
        "## Numeric Analysis",
        "",
        f"- Mean: `{format_number(stats.get('mean'))}`",
        f"- Std: `{format_number(stats.get('std'))}`",
        f"- Min/Max: `{format_number(stats.get('min'))}` / `{format_number(stats.get('max'))}`",
        f"- Original missing ratio: `{stats.get('original_missing_ratio'):.2%}`",
        "",
        "## Spatial Analysis",
        "",
        f"- Hotspot count: `{hotspot.get('hotspot_count')}`",
        f"- Connected high-value domains: `{hotspot.get('connected_domains')}`",
        f"- Gradient strength: `{format_number(gradient.get('gradient_strength'))}`",
        f"- Boundary change: `{format_number(gradient.get('boundary_change'))}`",
        "",
        "## Temporal Analysis",
        "",
        f"- Trend: `{temporal['trend'].get('trend')}`",
        f"- Slope: `{format_number(temporal['trend'].get('slope'))}`",
        f"- Change points: `{len(temporal.get('change_points', []))}`",
        "",
        "## Transformer Analysis",
        "",
        f"- Model: `{transformer.get('model', {}).get('name')}`",
        f"- Patch tokens: `{transformer.get('model', {}).get('token_count')}`",
        f"- Attention heads: `{transformer.get('model', {}).get('num_heads')}`",
        f"- Saliency p95: `{format_number(transformer.get('saliency_summary', {}).get('p95'))}`",
        "",
        "## Artifacts",
        "",
    ]
    for artifact in artifacts:
        lines.append(f"- `{artifact.get('type')}`: {artifact.get('path')}")
    if explanation.get("risks"):
        lines.extend(["", "## Risks", ""])
        lines.extend(f"- {risk}" for risk in explanation["risks"])
    return "\n".join(lines) + "\n"


def build_html_report(
    ctx: ExecutionContext,
    artifacts: Sequence[Mapping[str, Any]],
    explanation: Mapping[str, Any],
) -> str:
    artifact_items = "".join(
        f"<li><code>{escape(artifact.get('type', 'artifact'))}</code>: {escape(artifact.get('path', ''))}</li>"
        for artifact in artifacts
    )
    risks = "".join(f"<li>{escape(str(risk))}</li>" for risk in explanation.get("risks", []))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>GridVis Analysis Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 960px; margin: 24px auto; color: #1f2933; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>GridVis Analysis Report</h1>
  <p><strong>Task:</strong> {escape(get_task_type(ctx.plan))}</p>
  <p><strong>Region:</strong> {escape(ctx.region)}</p>
  <p><strong>Variable:</strong> {escape(str(ctx.variable))}</p>
  <h2>Summary</h2>
  <p>{escape(explanation.get('summary', ''))}</p>
  <h2>Artifacts</h2>
  <ul>{artifact_items}</ul>
  <h2>Risks</h2>
  <ul>{risks}</ul>
</body>
</html>
"""


def write_execution_log(ctx: ExecutionContext) -> Path:
    path = ctx.output_dir / "execution_log.json"
    path.write_text(json.dumps(ctx.logs, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def map_to_rgb(values: np.ndarray, *, diverging: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Map rendering expects a 2D array.")
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full((*arr.shape, 3), 235, dtype=np.uint8)

    if diverging:
        bound = float(np.nanmax(np.abs(arr[finite])))
        vmin, vmax = -bound, bound
        colors = DIVERGING_COLORS
    else:
        vmin = float(np.nanpercentile(arr[finite], 2))
        vmax = float(np.nanpercentile(arr[finite], 98))
        if vmax == vmin:
            vmax = vmin + 1.0
        colors = SEQUENTIAL_COLORS
    normalized = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    image = np.full((*arr.shape, 3), 230, dtype=np.uint8)
    image[finite] = interpolate_colors(normalized[finite], colors)
    return image


def interpolate_colors(values: np.ndarray, colors: Sequence[tuple[int, int, int]]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = values * (len(colors) - 1)
    low = np.floor(scale).astype(int)
    high = np.clip(low + 1, 0, len(colors) - 1)
    weight = (scale - low)[:, np.newaxis]
    low_colors = np.asarray([colors[idx] for idx in low], dtype=float)
    high_colors = np.asarray([colors[idx] for idx in high], dtype=float)
    return np.round(low_colors * (1 - weight) + high_colors * weight).astype(np.uint8)


def overlay_mask_boundary(image: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    if mask.shape != image.shape[:2] or not np.any(mask):
        return out
    boundary = ndimage.binary_dilation(mask) ^ ndimage.binary_erosion(mask)
    out[boundary] = color
    return out


def label_to_rgb(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    image = np.full((*labels.shape, 3), 238, dtype=np.uint8)
    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        image[labels == label_id] = palette[(label_id - 1) % len(palette)]
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG writer expects RGB image array.")
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def write_svg_heatmap(path: Path, image: np.ndarray, *, title: str) -> None:
    height, width, _ = image.shape
    png_path = path.with_suffix(".png")
    if not png_path.exists():
        write_png(png_path, image)
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + 32}" viewBox="0 0 {width} {height + 32}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="8" y="20" font-family="Arial, sans-serif" font-size="14" fill="#1f2933">{escape(title)}</text>
  <image x="0" y="32" width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def build_line_svg(series: Sequence[float], time_values: Sequence[str | None], *, title: str) -> str:
    values = np.asarray(series, dtype=float)
    width, height = 760, 320
    pad = 48
    finite = np.isfinite(values)
    if not finite.any():
        points = ""
        ymin, ymax = 0, 1
    else:
        ymin = float(np.nanmin(values[finite]))
        ymax = float(np.nanmax(values[finite]))
        if ymin == ymax:
            ymin -= 1
            ymax += 1
        xs = np.linspace(pad, width - pad, len(values))
        ys = height - pad - (values - ymin) / (ymax - ymin) * (height - 2 * pad)
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y, ok in zip(xs, ys, finite) if ok)
    start = time_values[0] if time_values else ""
    end = time_values[-1] if time_values else ""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{pad}" y="24" font-family="Arial, sans-serif" font-size="16" fill="#1f2933">{escape(title)}</text>
  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#475569"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#475569"/>
  <polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>
  <text x="{pad}" y="{height-12}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{escape(str(start))}</text>
  <text x="{width-pad-120}" y="{height-12}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{escape(str(end))}</text>
  <text x="8" y="{pad}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{ymax:.3g}</text>
  <text x="8" y="{height-pad}" font-family="Arial, sans-serif" font-size="11" fill="#475569">{ymin:.3g}</text>
</svg>
"""


def build_trajectory_svg(
    trajectory: Sequence[Mapping[str, Any]],
    shape: tuple[int, int],
    *,
    title: str,
) -> str:
    width, height = 760, 420
    pad = 40
    rows, cols = shape
    points = []
    for item in trajectory:
        y, x = item["centroid_index"]
        sx = pad + float(x) / max(1, cols - 1) * (width - 2 * pad)
        sy = pad + float(y) / max(1, rows - 1) * (height - 2 * pad)
        points.append((sx, sy, item.get("time_index")))
    poly = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
    circles = "".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#dc2626"><title>time {idx}</title></circle>'
        for x, y, idx in points
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{pad}" y="24" font-family="Arial, sans-serif" font-size="16" fill="#1f2933">{escape(title)}</text>
  <rect x="{pad}" y="{pad}" width="{width-2*pad}" height="{height-2*pad}" fill="#f8fafc" stroke="#cbd5e1"/>
  <polyline points="{poly}" fill="none" stroke="#dc2626" stroke-width="2"/>
  {circles}
</svg>
"""


def build_scatter_svg(primary: np.ndarray, secondary: np.ndarray, *, xlabel: str, ylabel: str) -> str:
    width, height = 760, 420
    pad = 48
    if primary.size == 0:
        points = ""
        xmin = xmax = ymin = ymax = 0
    else:
        xmin, xmax = float(np.nanmin(primary)), float(np.nanmax(primary))
        ymin, ymax = float(np.nanmin(secondary)), float(np.nanmax(secondary))
        if xmin == xmax:
            xmax = xmin + 1
        if ymin == ymax:
            ymax = ymin + 1
        xs = pad + (primary - xmin) / (xmax - xmin) * (width - 2 * pad)
        ys = height - pad - (secondary - ymin) / (ymax - ymin) * (height - 2 * pad)
        points = "".join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.6" fill="#2563eb" opacity="0.38"/>' for x, y in zip(xs, ys))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{pad}" y="24" font-family="Arial, sans-serif" font-size="16" fill="#1f2933">{escape(xlabel)} vs {escape(ylabel)}</text>
  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#475569"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#475569"/>
  {points}
  <text x="{width/2-40}" y="{height-12}" font-family="Arial, sans-serif" font-size="12" fill="#475569">{escape(xlabel)}</text>
  <text x="8" y="{height/2}" font-family="Arial, sans-serif" font-size="12" fill="#475569">{escape(ylabel)}</text>
</svg>
"""


def prepare_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is None:
        output_dir = MODULES_DIR / "05_results_output" / time.strftime("run_%Y%m%d_%H%M%S")
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_task_type(plan: Mapping[str, Any]) -> str:
    return str(plan.get("task_graph", {}).get("task_type", "distribution"))


def nanmean_time(data: np.ndarray) -> np.ndarray:
    count = np.sum(np.isfinite(data), axis=0)
    return np.divide(
        np.nansum(data, axis=0),
        count,
        out=np.full(data.shape[1:], np.nan, dtype=float),
        where=count > 0,
    )


def spatial_mean_series(data: np.ndarray) -> np.ndarray:
    series = []
    for tidx in range(data.shape[0]):
        values = data[tidx]
        finite = np.isfinite(values)
        series.append(float(np.nanmean(values)) if np.any(finite) else np.nan)
    return np.asarray(series, dtype=float)


def fill_nan_nearest(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    missing = ~np.isfinite(arr)
    if not missing.any():
        return arr
    valid = ~missing
    if not valid.any():
        return np.zeros_like(arr)
    _, indices = ndimage.distance_transform_edt(missing, return_indices=True)
    arr[missing] = arr[tuple(index[missing] for index in indices)]
    return arr


def strip_large_arrays(value: Any) -> Any:
    if isinstance(value, Mapping):
        output = {}
        for key, item in value.items():
            if key == "patch_tokens" and isinstance(item, list):
                output["patch_token_count"] = len(item)
                output["patch_tokens_preview"] = [strip_token_vector(token) for token in item[:5]]
                continue
            if key == "top_salient_patches" and isinstance(item, list):
                output[key] = [strip_token_vector(token) for token in item[:10]]
                continue
            if key == "global_token" and isinstance(item, list):
                output["global_token_dim"] = len(item)
                output["global_token_preview"] = item[:8]
                continue
            if isinstance(item, np.ndarray):
                output[key] = {
                    "shape": list(item.shape),
                    "min": safe_float(np.nanmin(item)) if item.size and np.isfinite(item).any() else None,
                    "max": safe_float(np.nanmax(item)) if item.size and np.isfinite(item).any() else None,
                }
            else:
                output[key] = strip_large_arrays(item)
        return output
    if isinstance(value, list):
        return [strip_large_arrays(item) for item in value]
    return value


def strip_token_vector(token: Any) -> Any:
    if not isinstance(token, Mapping):
        return token
    compact = {key: value for key, value in token.items() if key != "vector"}
    if "vector" in token:
        compact["vector_dim"] = len(token.get("vector") or [])
        compact["vector_preview"] = (token.get("vector") or [])[:4]
    return strip_large_arrays(compact)


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_ratio(part: int, total: int) -> float:
    return float(part / total) if total else 0.0


def format_number(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "NA"
    return f"{number:.3g}"


def unique(values: Sequence[Any]) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        marker = str(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def load_plan(path: str | Path | None, args: argparse.Namespace) -> dict[str, Any]:
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "task_graph": {
            "task_type": args.task_type,
            "region": args.region,
            "variables": [args.variable] if args.variable else [],
            "time_range": args.time_range,
        },
        "visualization_strategy": {
            "chart_type": args.visualization_strategy,
            "layout": args.layout,
            "color_encoding": args.color_encoding,
            "focus_region": args.region,
            "explanation_template": args.explanation_template,
        },
        "uncertainty_risks": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute GridVis visualization plans and generate analysis reports.")
    parser.add_argument("--plan-json", help="Path to Step 3 plan JSON.")
    parser.add_argument("--source", required=True, help="Raw grid source accepted by Step 1.")
    parser.add_argument("--visualization_strategy", default="heatmap + contour overlay", help="Chart strategy when --plan-json is not supplied.")
    parser.add_argument("--task_type", default="distribution", help="Task type when --plan-json is not supplied.")
    parser.add_argument("--region", default="数据覆盖区域", help="Target region.")
    parser.add_argument("--variable", default=None, help="Target variable.")
    parser.add_argument("--time_range", default="数据时间范围", help="Target time range.")
    parser.add_argument("--layout", default="single map with summary cards", help="Layout hint.")
    parser.add_argument("--color_encoding", default="sequential color scale", help="Color encoding hint.")
    parser.add_argument("--explanation_template", default="", help="Explanation template from Step 3.")
    parser.add_argument("--output_format", default="png", choices=("png", "svg", "html"), help="Preferred chart output format.")
    parser.add_argument("--output_dir", help="Output directory. Defaults to 05_results_output/run_*.")
    parser.add_argument("--no-fill", action="store_true", help="Do not fill missing data in Step 1.")
    args = parser.parse_args()

    plan = load_plan(args.plan_json, args)
    variables = [args.variable] if args.variable else None
    result = execute_plan(
        plan,
        args.source,
        output_dir=args.output_dir,
        output_format=args.output_format,
        variables=variables,
        fill_missing=not args.no_fill,
    )
    print(json.dumps(json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
