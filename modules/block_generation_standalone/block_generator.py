from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


GRID_PATTERN_LABELS = {
    "H": {"name": "高值热点型", "en": "Hotspot", "color": "#d94841"},
    "L": {"name": "低值冷点型", "en": "Lowspot", "color": "#2f6fdd"},
    "G": {"name": "梯度过渡型", "en": "Gradient", "color": "#2a9d72"},
    "B": {"name": "边界突变型", "en": "Boundary", "color": "#f08a24"},
    "D": {"name": "扩散分散型", "en": "Diffuse", "color": "#8b5bd6"},
    "U": {"name": "均匀稳定型", "en": "Uniform", "color": "#7b8794"},
    "M": {"name": "混合复杂型", "en": "Mixed", "color": "#334155"},
    "N": {"name": "噪声不确定型", "en": "Noisy / Uncertain", "color": "#b45309"},
}


DEFAULT_THRESHOLDS = {
    "missing_ratio_noisy": 0.30,
    "outlier_ratio_noisy": 0.18,
    "uniform_std_norm": 0.10,
    "uniform_gradient_norm": 0.05,
    "hotspot_ratio": 0.25,
    "lowspot_ratio": 0.25,
    "compact_components": 2,
    "gradient_mean_norm": 0.40,
    "gradient_edge_max_for_gradient": 0.30,
    "edge_strength_norm": 0.50,
    "diffuse_components": 4,
    "mixed_score_gap": 0.12,
}


def generate_blocks(
    grid: np.ndarray,
    *,
    rows: int = 9,
    cols: int = 14,
    block_shape: tuple[int, int] | None = None,
    mask: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Generate semantic blocks from a 2D grid or a 3D time stack.

    `grid` accepts:
    - 2D: `[y, x]`
    - 3D: `[time, y, x]`, where the latest slice drives the label and the
      whole stack supplies mean/trend/volatility context.
    """

    stack = normalize_grid(grid)
    latest = stack[-1]
    mean_map = nanmean_axis0(stack)
    trend_map = compute_trend_map(stack)
    anomaly_map = latest - mean_map
    valid_mask = np.isfinite(mean_map) if mask is None else np.asarray(mask, dtype=bool) & np.isfinite(mean_map)

    height, width = latest.shape
    lat_values = np.asarray(lat if lat is not None else np.arange(height), dtype=float)
    lon_values = np.asarray(lon if lon is not None else np.arange(width), dtype=float)
    if lat_values.size != height or lon_values.size != width:
        raise ValueError("lat/lon length must match grid height/width")

    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    slices = block_slices(height, width, rows=rows, cols=cols, block_shape=block_shape)
    gy, gx, gradient = gradient_layers(latest, valid_mask)

    global_values = latest[np.isfinite(latest) & valid_mask]
    if global_values.size == 0:
        global_values = mean_map[np.isfinite(mean_map) & valid_mask]
    if global_values.size == 0:
        global_values = np.array([0.0])

    global_min = float(np.nanmin(global_values))
    global_max = float(np.nanmax(global_values))
    global_range = max(global_max - global_min, 1e-9)
    high_threshold = float(np.nanpercentile(global_values, 75))
    low_threshold = float(np.nanpercentile(global_values, 25))
    median = float(np.nanmedian(global_values))
    mad = float(np.nanmedian(np.abs(global_values - median)))
    outlier_scale = max(1.4826 * mad, float(np.nanstd(global_values)), 1e-9)
    valid_gradient = gradient[np.isfinite(latest) & valid_mask]
    gradient_p95 = max(float(np.nanpercentile(valid_gradient, 95)) if valid_gradient.size else 1.0, 1e-9)

    blocks: list[dict[str, Any]] = []
    label_matrix: list[list[str | None]] = [[None for _ in range(slices["cols"])] for _ in range(slices["rows"])]
    confidence_matrix = np.full((slices["rows"], slices["cols"]), np.nan, dtype=float)
    saliency_matrix = np.full((slices["rows"], slices["cols"]), np.nan, dtype=float)

    for row, col, y0, y1, x0, x1 in slices["items"]:
        block_mask = valid_mask[y0:y1, x0:x1]
        if not block_mask.any():
            continue

        local = latest[y0:y1, x0:x1]
        local_valid = block_mask & np.isfinite(local)
        valid_cells = int(np.count_nonzero(local_valid))
        mask_cells = int(np.count_nonzero(block_mask))
        if valid_cells == 0:
            continue

        values = local[local_valid]
        high_binary = (local >= high_threshold) & local_valid
        low_binary = (local <= low_threshold) & local_valid
        high_components, high_largest = connected_component_stats(high_binary)
        low_components, low_largest = connected_component_stats(low_binary)
        largest_component = max(high_largest, low_largest)
        gradient_values = gradient[y0:y1, x0:x1][local_valid]
        outliers = np.abs((values - median) / outlier_scale) > 3.5
        block_series = np.array(
            [safe_nanmean(time_slice[y0:y1, x0:x1][block_mask]) for time_slice in stack],
            dtype=float,
        )
        valid_series = block_series[np.isfinite(block_series)]

        value_min = float(np.nanmin(values))
        value_max = float(np.nanmax(values))
        value_mean = float(np.nanmean(values))
        value_std = float(np.nanstd(values))
        gradient_mean = float(np.nanmean(gradient_values)) if gradient_values.size else 0.0
        gradient_std = float(np.nanstd(gradient_values)) if gradient_values.size else 0.0
        edge_strength = float(np.nanpercentile(gradient_values, 90)) if gradient_values.size else 0.0

        features = {
            "mean": safe_float(value_mean),
            "std": safe_float(value_std),
            "min": safe_float(value_min),
            "max": safe_float(value_max),
            "q25": safe_float(np.nanpercentile(values, 25)),
            "q50": safe_float(np.nanpercentile(values, 50)),
            "q75": safe_float(np.nanpercentile(values, 75)),
            "value_range": safe_float(value_max - value_min),
            "gradient_mean": safe_float(gradient_mean),
            "gradient_std": safe_float(gradient_std),
            "dominant_gradient_direction": dominant_gradient_direction(gx[y0:y1, x0:x1][local_valid], gy[y0:y1, x0:x1][local_valid]),
            "edge_strength": safe_float(edge_strength),
            "hotspot_ratio": safe_float(np.count_nonzero(high_binary) / max(valid_cells, 1), 4),
            "lowspot_ratio": safe_float(np.count_nonzero(low_binary) / max(valid_cells, 1), 4),
            "connected_components_high": high_components,
            "connected_components_low": low_components,
            "largest_component_ratio": safe_float(largest_component / max(valid_cells, 1), 4),
            "moran_i": simple_moran_i(np.where(block_mask, local, np.nan)),
            "missing_ratio": safe_float(1.0 - valid_cells / max(mask_cells, 1), 4),
            "outlier_ratio": safe_float(np.count_nonzero(outliers) / max(valid_cells, 1), 4),
            "std_norm": safe_float(value_std / global_range, 4),
            "gradient_mean_norm": safe_float(gradient_mean / gradient_p95, 4),
            "edge_strength_norm": safe_float(edge_strength / gradient_p95, 4),
            "peak_prominence_norm": safe_float(max(0.0, value_max - value_mean) / global_range, 4),
            "low_prominence_norm": safe_float(max(0.0, value_mean - value_min) / global_range, 4),
        }

        scores = label_scores(features, thresholds)
        primary_label, confidence = choose_semantic_label(features, scores, thresholds)
        secondary_labels = [
            label
            for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if label not in {primary_label, "M", "N"} and score >= 0.45
        ][:2]
        evidence = semantic_evidence(primary_label, features)
        saliency = max(scores.get(label, 0) for label in ("H", "L", "G", "B", "D"))
        trend = safe_nanmean(trend_map[y0:y1, x0:x1][block_mask])

        block = {
            "block_id": f"B_{row:03d}_{col:03d}",
            "id": f"B_{row:03d}_{col:03d}",
            "row": row,
            "col": col,
            "index_range": {"y": [int(y0), int(y1)], "x": [int(x0), int(x1)]},
            "bbox": {
                "lat_min": safe_float(lat_values[y0], 4),
                "lat_max": safe_float(lat_values[y1 - 1], 4),
                "lon_min": safe_float(lon_values[x0], 4),
                "lon_max": safe_float(lon_values[x1 - 1], 4),
            },
            "center": {
                "lat": safe_float(float(np.mean(lat_values[y0:y1])), 4),
                "lon": safe_float(float(np.mean(lon_values[x0:x1])), 4),
            },
            "primary_label": primary_label,
            "primary_label_name": GRID_PATTERN_LABELS[primary_label]["name"],
            "secondary_labels": secondary_labels,
            "confidence": safe_float(confidence, 4),
            "scores": scores,
            "features": features,
            "evidence": evidence,
            "llm_explanation": semantic_explanation(primary_label, evidence),
            "mean": safe_nanmean(mean_map[y0:y1, x0:x1][local_valid]),
            "latest": safe_nanmean(values),
            "trend": trend,
            "anomaly": safe_nanmean(np.abs(anomaly_map[y0:y1, x0:x1][local_valid])),
            "volatility": safe_float(float(np.std(valid_series))) if valid_series.size else None,
            "finite_ratio": safe_float(valid_cells / max(mask_cells, 1), 4),
            "saliency": safe_float(saliency, 4),
            "pattern_type": primary_label,
            "trend_label": "rising" if trend and trend > 1e-5 else "falling" if trend and trend < -1e-5 else "stable",
            "uncertainty": "high" if primary_label == "N" else "medium" if scores.get("N", 0) > 0.45 else "low",
            "llm_tokens": [
                f"label:{primary_label}",
                f"confidence:{safe_float(confidence, 4)}",
                f"risk:{'high' if primary_label == 'N' else 'low'}",
            ],
        }
        blocks.append(block)
        label_matrix[row][col] = primary_label
        confidence_matrix[row, col] = confidence
        saliency_matrix[row, col] = saliency

    blocks.sort(key=lambda item: item["saliency"] or 0, reverse=True)
    label_distribution = {label: 0 for label in GRID_PATTERN_LABELS}
    for block in blocks:
        label_distribution[block["primary_label"]] += 1

    return {
        "grid": {"rows": slices["rows"], "cols": slices["cols"], "height": height, "width": width},
        "label_matrix": label_matrix,
        "confidence_matrix": json_matrix(confidence_matrix),
        "matrix": json_matrix(saliency_matrix),
        "blocks": blocks,
        "top_blocks": blocks[:12],
        "label_distribution": label_distribution,
        "labels": GRID_PATTERN_LABELS,
        "thresholds": thresholds,
        "source_note": {
            "label": "Pattern Label generated by Block Semantic Extractor.",
            "explanation": "Explanation generated from algorithmic evidence.",
            "combined": "Algorithmic Label + Evidence Explanation.",
        },
        "generation_steps": [
            "1. 将二维网格或时序栈规范化为 [time, y, x]",
            "2. 按 rows/cols 或 block_shape 生成每个 block 的 y/x index range",
            "3. 对每个 block 提取统计、梯度、边界、连通域、缺失率和异常点比例",
            "4. 根据可复现规则计算 H/L/G/B/D/U/M/N 分数",
            "5. 选择 primary_label、secondary_labels，并输出 evidence",
        ],
    }


def block_slices(
    height: int,
    width: int,
    *,
    rows: int,
    cols: int,
    block_shape: tuple[int, int] | None,
) -> dict[str, Any]:
    if block_shape is not None:
        block_h, block_w = block_shape
        if block_h <= 0 or block_w <= 0:
            raise ValueError("block_shape must be positive")
        y_edges = list(range(0, height, block_h)) + [height]
        x_edges = list(range(0, width, block_w)) + [width]
        y_edges = sorted(set(y_edges))
        x_edges = sorted(set(x_edges))
    else:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows/cols must be positive")
        y_edges = np.linspace(0, height, rows + 1, dtype=int).tolist()
        x_edges = np.linspace(0, width, cols + 1, dtype=int).tolist()

    items = []
    for row in range(len(y_edges) - 1):
        for col in range(len(x_edges) - 1):
            y0, y1 = y_edges[row], y_edges[row + 1]
            x0, x1 = x_edges[col], x_edges[col + 1]
            if y1 > y0 and x1 > x0:
                items.append((row, col, y0, y1, x0, x1))
    return {"rows": len(y_edges) - 1, "cols": len(x_edges) - 1, "items": items}


def normalize_grid(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=float)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError("grid must be 2D [y, x] or 3D [time, y, x]")
    return arr


def gradient_layers(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(values) & mask
    fallback = float(np.nanmean(values[finite])) if finite.any() else 0.0
    filled = np.where(finite, values, fallback)
    gy, gx = np.gradient(filled)
    gradient = np.sqrt(gx * gx + gy * gy)
    return gy, gx, gradient


def connected_component_stats(binary: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(binary, dtype=bool)
    visited = np.zeros(arr.shape, dtype=bool)
    count = 0
    largest = 0

    for start_y, start_x in zip(*np.where(arr & ~visited)):
        if visited[start_y, start_x]:
            continue
        count += 1
        size = 0
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        while queue:
            y, x = queue.popleft()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < arr.shape[0] and 0 <= nx < arr.shape[1] and arr[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        largest = max(largest, size)
    return count, largest


def label_scores(features: dict[str, Any], thresholds: dict[str, float]) -> dict[str, float]:
    std_norm = float(features.get("std_norm") or 0)
    gradient_norm = float(features.get("gradient_mean_norm") or 0)
    edge_norm = float(features.get("edge_strength_norm") or 0)
    hotspot_ratio = float(features.get("hotspot_ratio") or 0)
    lowspot_ratio = float(features.get("lowspot_ratio") or 0)
    high_components = int(features.get("connected_components_high") or 0)
    low_components = int(features.get("connected_components_low") or 0)
    largest_ratio = float(features.get("largest_component_ratio") or 0)
    missing_ratio = float(features.get("missing_ratio") or 0)
    outlier_ratio = float(features.get("outlier_ratio") or 0)
    peak_prominence = float(features.get("peak_prominence_norm") or 0)
    low_prominence = float(features.get("low_prominence_norm") or 0)

    compact_limit = max(1.0, thresholds["compact_components"])
    diffuse_limit = max(1.0, thresholds["diffuse_components"])
    compact_high = clamp01(1.0 - max(0, high_components - 1) / compact_limit)
    compact_low = clamp01(1.0 - max(0, low_components - 1) / compact_limit)
    fragmented = clamp01(max(high_components, low_components) / diffuse_limit) * clamp01(1.0 - largest_ratio)

    scores = {
        "N": max(
            clamp01(missing_ratio / thresholds["missing_ratio_noisy"]),
            clamp01(outlier_ratio / thresholds["outlier_ratio_noisy"]),
        ),
        "U": clamp01(1.0 - max(
            std_norm / thresholds["uniform_std_norm"],
            gradient_norm / thresholds["uniform_gradient_norm"],
        )),
        "H": clamp01(
            0.42 * clamp01(hotspot_ratio / thresholds["hotspot_ratio"])
            + 0.28 * compact_high
            + 0.20 * clamp01(largest_ratio / 0.35)
            + 0.10 * peak_prominence
        ),
        "L": clamp01(
            0.42 * clamp01(lowspot_ratio / thresholds["lowspot_ratio"])
            + 0.28 * compact_low
            + 0.20 * clamp01(largest_ratio / 0.35)
            + 0.10 * low_prominence
        ),
        "G": clamp01(
            0.72 * clamp01(gradient_norm / thresholds["gradient_mean_norm"])
            + 0.28 * clamp01(1.0 - edge_norm / max(thresholds["gradient_edge_max_for_gradient"], 1e-6))
        ),
        "B": clamp01(edge_norm / thresholds["edge_strength_norm"]),
        "D": clamp01(0.72 * fragmented + 0.28 * clamp01((high_components + low_components) / (2.0 * diffuse_limit))),
        "M": 0.0,
    }
    ranked = sorted([scores[key] for key in ("H", "L", "G", "B", "D", "U")], reverse=True)
    gap = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
    scores["M"] = clamp01((1.0 - gap / max(thresholds["mixed_score_gap"], 1e-6)) * 0.55 + ranked[1] * 0.45)
    return {key: safe_float(value, 4) or 0.0 for key, value in scores.items()}


def choose_semantic_label(features: dict[str, Any], scores: dict[str, float], thresholds: dict[str, float]) -> tuple[str, float]:
    if (features.get("missing_ratio") or 0) > thresholds["missing_ratio_noisy"] or scores.get("N", 0) >= 0.92:
        return "N", scores.get("N", 0)
    if (
        (features.get("std_norm") or 0) < thresholds["uniform_std_norm"]
        and (features.get("gradient_mean_norm") or 0) < thresholds["uniform_gradient_norm"]
    ):
        return "U", max(scores.get("U", 0), 0.72)

    ordered = sorted(
        ((label, score) for label, score in scores.items() if label not in {"M", "N", "U"}),
        key=lambda item: item[1],
        reverse=True,
    )
    top_label, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    if top_score < 0.42 or (top_score - second_score) < thresholds["mixed_score_gap"]:
        return "M", max(scores.get("M", 0), min(0.86, 0.58 + second_score * 0.25))
    return top_label, top_score


def semantic_evidence(label: str, features: dict[str, Any]) -> list[str]:
    if label == "H":
        evidence = [
            f"高值像元占比为 {round((features.get('hotspot_ratio') or 0) * 100, 1)}%",
            f"高值连通域数量为 {features.get('connected_components_high')}",
            "最大值显著高于 block 均值",
            f"最大连通域面积占比为 {round((features.get('largest_component_ratio') or 0) * 100, 1)}%",
        ]
    elif label == "L":
        evidence = [
            f"低值像元占比为 {round((features.get('lowspot_ratio') or 0) * 100, 1)}%",
            f"低值连通域数量为 {features.get('connected_components_low')}",
            "最小值显著低于 block 均值",
            f"最大连通域面积占比为 {round((features.get('largest_component_ratio') or 0) * 100, 1)}%",
        ]
    elif label == "G":
        evidence = [
            f"平均梯度强度归一化值为 {round(features.get('gradient_mean_norm') or 0, 3)}",
            f"主导梯度方向为 {features.get('dominant_gradient_direction') or 'flat'}",
            "空间变化连续，未形成单一强热点中心",
        ]
    elif label == "B":
        evidence = [
            f"边界强度归一化值为 {round(features.get('edge_strength_norm') or 0, 3)}",
            "梯度集中于局部突变带",
            f"block 内极差为 {features.get('value_range')}",
        ]
    elif label == "D":
        evidence = [
            f"高值连通域数量为 {features.get('connected_components_high')}",
            f"低值连通域数量为 {features.get('connected_components_low')}",
            f"最大连通域面积占比为 {round((features.get('largest_component_ratio') or 0) * 100, 1)}%",
            "斑块数量较多且空间分布较破碎",
        ]
    elif label == "U":
        evidence = [
            f"标准差归一化值为 {round(features.get('std_norm') or 0, 3)}",
            f"平均梯度强度归一化值为 {round(features.get('gradient_mean_norm') or 0, 3)}",
            "局部极值和边界突变均不明显",
        ]
    elif label == "N":
        evidence = [
            f"缺失率为 {round((features.get('missing_ratio') or 0) * 100, 1)}%",
            f"异常点比例为 {round((features.get('outlier_ratio') or 0) * 100, 1)}%",
            "当前 block 的空间结构判断可靠性较低",
        ]
    else:
        evidence = [
            "多个模式得分接近，单一标签区分度不足",
            f"高值占比 {round((features.get('hotspot_ratio') or 0) * 100, 1)}%，低值占比 {round((features.get('lowspot_ratio') or 0) * 100, 1)}%",
            f"边界强度归一化值为 {round(features.get('edge_strength_norm') or 0, 3)}",
        ]

    if (features.get("missing_ratio") or 0) <= 0.05:
        evidence.append("缺失率较低，数据质量较好")
    return evidence


def semantic_explanation(label: str, evidence: list[str]) -> str:
    lead = {
        "H": "该 block 被识别为高值热点型，说明内部存在相对集中的局部高值中心。",
        "L": "该 block 被识别为低值冷点型，说明内部存在相对集中的局部低值中心。",
        "G": "该 block 被识别为梯度过渡型，说明数值沿某个方向呈连续变化。",
        "B": "该 block 被识别为边界突变型，说明内部或边缘存在明显突变带。",
        "D": "该 block 被识别为扩散分散型，说明异常斑块较多且不集中于单一中心。",
        "U": "该 block 被识别为均匀稳定型，说明内部空间变化较弱。",
        "M": "该 block 被识别为混合复杂型，说明多个空间结构同时存在且得分接近。",
        "N": "该 block 被识别为噪声不确定型，说明当前数据质量或稳定性不足以支持强判断。",
    }[label]
    return f"{lead} 主要证据包括：{'；'.join(evidence[:3])}。"


def dominant_gradient_direction(gx: np.ndarray, gy: np.ndarray) -> str | None:
    valid = np.isfinite(gx) & np.isfinite(gy)
    if np.count_nonzero(valid) == 0:
        return None
    mean_x = float(np.nanmean(gx[valid]))
    mean_y = float(np.nanmean(gy[valid]))
    magnitude = math.hypot(mean_x, mean_y)
    if magnitude < 1e-12:
        return "flat"
    angle = (math.degrees(math.atan2(mean_y, mean_x)) + 360.0) % 360.0
    for upper, label in (
        (22.5, "E"),
        (67.5, "NE"),
        (112.5, "N"),
        (157.5, "NW"),
        (202.5, "W"),
        (247.5, "SW"),
        (292.5, "S"),
        (337.5, "SE"),
        (360.0, "E"),
    ):
        if angle < upper:
            return label
    return "E"


def simple_moran_i(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if np.count_nonzero(finite) < 4:
        return None
    mean = float(np.nanmean(arr[finite]))
    centered = np.where(finite, arr - mean, np.nan)
    denominator = float(np.nansum(centered * centered))
    if denominator <= 1e-12:
        return None
    numerator = 0.0
    weights = 0
    for dy, dx in ((1, 0), (0, 1)):
        left = centered[:-dy or None, :-dx or None]
        right = centered[dy:, dx:]
        valid = np.isfinite(left) & np.isfinite(right)
        if np.count_nonzero(valid):
            numerator += float(np.nansum(left[valid] * right[valid])) * 2.0
            weights += int(np.count_nonzero(valid)) * 2
    if weights == 0:
        return None
    return safe_float((np.count_nonzero(finite) / weights) * (numerator / denominator), 4)


def compute_trend_map(stack: np.ndarray) -> np.ndarray:
    if stack.shape[0] == 1:
        return np.zeros(stack.shape[1:], dtype=float)
    x = np.arange(stack.shape[0], dtype=float)
    x = x - x.mean()
    denominator = float(np.sum(x * x)) or 1.0
    mean = nanmean_axis0(stack)
    centered = np.where(np.isfinite(stack), stack - mean[np.newaxis, :, :], 0)
    return np.tensordot(x, centered, axes=(0, 0)) / denominator


def nanmean_axis0(stack: np.ndarray) -> np.ndarray:
    valid = np.isfinite(stack)
    counts = valid.sum(axis=0)
    total = np.where(valid, stack, 0).sum(axis=0)
    return np.divide(total, counts, out=np.full(stack.shape[1:], np.nan, dtype=float), where=counts > 0)


def safe_nanmean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return safe_float(float(np.mean(finite)))


def safe_float(value: Any, digits: int = 6) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, digits)


def clamp01(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def json_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    return [[safe_float(value, 4) for value in row] for row in matrix]


def load_grid(path: str | Path) -> np.ndarray:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        return np.load(source)
    if suffix == ".csv":
        return np.genfromtxt(source, delimiter=",")
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("data") or payload.get("grid") or payload.get("values")
        return np.asarray(payload, dtype=float)
    raise ValueError("supported input formats: .npy, .csv, .json")


def demo_grid(height: int = 72, width: int = 112, time_steps: int = 5) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    base = 20 + 12 * (x / max(width - 1, 1)) + 6 * np.sin(y / 7.0)
    hotspot = 34 * np.exp(-(((x - width * 0.72) ** 2) / (2 * 9**2) + ((y - height * 0.32) ** 2) / (2 * 7**2)))
    lowspot = -24 * np.exp(-(((x - width * 0.28) ** 2) / (2 * 8**2) + ((y - height * 0.68) ** 2) / (2 * 8**2)))
    boundary = np.where(x > width * 0.48, 14, 0)
    diffuse = np.where(((x // 9 + y // 8) % 5) == 0, 10, 0)
    latest = base + hotspot + lowspot + boundary + diffuse
    stack = []
    for t in range(time_steps):
        stack.append(latest + t * (0.5 + y / max(height, 1) * 0.3) + np.sin((x + t) / 11.0))
    arr = np.asarray(stack, dtype=float)
    arr[:, 5:13, 5:24] = np.nan
    return arr


def parse_shape(value: str) -> tuple[int, int]:
    if "x" in value:
        left, right = value.lower().split("x", 1)
    elif "," in value:
        left, right = value.split(",", 1)
    else:
        left = right = value
    return int(left), int(right)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone block generation demo.")
    parser.add_argument("source", nargs="?", help="Optional .npy/.csv/.json grid. Omit to use a synthetic demo grid.")
    parser.add_argument("--rows", type=int, default=9, help="Block rows when --block-shape is not used.")
    parser.add_argument("--cols", type=int, default=14, help="Block columns when --block-shape is not used.")
    parser.add_argument("--block-shape", type=parse_shape, default=None, help="Cell size per block, e.g. 16x16.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    parser.add_argument("--top", type=int, default=5, help="How many top blocks to print in compact mode.")
    args = parser.parse_args()

    grid = load_grid(args.source) if args.source else demo_grid()
    result = generate_blocks(grid, rows=args.rows, cols=args.cols, block_shape=args.block_shape)

    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
        return

    compact = {
        "grid": result["grid"],
        "label_distribution": result["label_distribution"],
        "label_matrix": result["label_matrix"],
        "top_blocks": result["top_blocks"][: args.top],
        "generation_steps": result["generation_steps"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
