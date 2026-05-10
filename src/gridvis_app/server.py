from __future__ import annotations

import gzip
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import xarray as xr
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULES_DIR = PROJECT_ROOT / "modules"
LLM_CORE_DIR = MODULES_DIR / "03_llm_core"
if str(LLM_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_CORE_DIR))
DATA_DIR = PROJECT_ROOT / "data" / "2000"
CACHE_DIR = PROJECT_ROOT / ".gridvis_cache"
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "final"
FRONTEND_FILE = FRONTEND_DIR / "index.html"
CHINA_GEOJSON = PROJECT_ROOT / "assets" / "china.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = int(os.getenv("GRIDVIS_PORT", "8787"))
MAX_GRID_POINTS = 180
ANALYSIS_CACHE_VERSION = "v7-grid-pattern-semantics"

REGION_BOUNDS = {
    "china": None,
    "all": None,
    "north_china": (105.0, 125.0, 34.0, 43.5),
    "jingjinji": (113.0, 120.5, 36.0, 42.5),
    "east_china": (114.0, 123.5, 24.0, 38.5),
    "south_china": (105.0, 122.5, 18.0, 26.5),
    "northeast": (118.0, 135.5, 38.0, 54.5),
    "southwest": (97.0, 112.5, 21.0, 35.5),
    "northwest": (73.0, 112.5, 32.0, 50.5),
    "yangtze_delta": (116.0, 123.5, 28.0, 34.5),
    "pearl_delta": (111.0, 116.0, 21.0, 24.5),
}

REGION_LABELS = {
    "china": "中国全域",
    "all": "数据覆盖区域",
    "north_china": "华北地区",
    "jingjinji": "京津冀",
    "east_china": "华东地区",
    "south_china": "华南地区",
    "northeast": "东北地区",
    "southwest": "西南地区",
    "northwest": "西北地区",
    "yangtze_delta": "长三角",
    "pearl_delta": "珠三角",
}

VARIABLE_ALIAS = {
    "pm2.5": "PM2.5",
    "pm25": "PM2.5",
    "pm 2.5": "PM2.5",
    "污染": "PM2.5",
    "颗粒物": "PM2.5",
    "温度": "temp",
    "气温": "temp",
    "temperature": "temp",
    "湿度": "rhum",
    "相对湿度": "rhum",
    "humidity": "rhum",
    "风": "wind",
    "风速": "wind",
    "wind": "wind",
    "降水": "prec",
    "降雨": "rain",
    "precipitation": "prec",
    "气压": "pres",
    "pressure": "pres",
    "高程": "elevation",
    "地形": "elevation",
    "elevation": "elevation",
    "长波": "lrad",
    "longwave": "lrad",
    "短波": "srad",
    "辐射": "srad",
    "shortwave": "srad",
    "比湿": "shum",
    "specific humidity": "shum",
    "降雪": "snow",
    "snow": "snow",
    "bias-corrected precipitation": "bcpr",
    "bcpr": "bcpr",
}


@dataclass(frozen=True)
class Catalog:
    files: tuple[Path, ...]
    variables: tuple[str, ...]
    variable_units: dict[str, str | None]
    variable_names: dict[str, str | None]
    lat: np.ndarray
    lon: np.ndarray
    dates: tuple[str, ...]
    fingerprint: str


_CATALOG: Catalog | None = None


def safe_float(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return safe_float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def parse_date(path: Path) -> str:
    match = re.search(r"(\d{8})", path.stem)
    if not match:
        return path.stem
    raw = match.group(1)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def extract_query_date(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", text)
    if not match:
        match = re.search(r"(20\d{2})(\d{2})(\d{2})", text)
    if not match:
        return None
    year, month, day = match.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def select_date_index(dates: tuple[str, ...], requested: str | None) -> tuple[int, str]:
    if not dates:
        raise ValueError("Catalog contains no dates.")
    normalized = extract_query_date(requested) or requested
    if normalized:
        normalized = str(normalized).strip()
        if normalized in dates:
            idx = dates.index(normalized)
            return idx, dates[idx]
    return len(dates) - 1, dates[-1]


def get_catalog() -> Catalog:
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG

    files = tuple(sorted(DATA_DIR.glob("*.nc")))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {DATA_DIR}")

    with xr.open_dataset(files[0], decode_times=True) as ds:
        lat = np.asarray(ds["lat"].values, dtype=float)
        lon = np.asarray(ds["lon"].values, dtype=float)
        variables = tuple(str(name) for name in ds.data_vars)
        units = {name: ds[name].attrs.get("units") for name in variables}
        long_names = {name: ds[name].attrs.get("long_name") for name in variables}

    fingerprint_items = [
        str(len(files)),
        str(int(max(path.stat().st_mtime for path in files))),
        str(files[0].stat().st_size),
        str(files[-1].stat().st_size),
    ]
    _CATALOG = Catalog(
        files=files,
        variables=variables,
        variable_units=units,
        variable_names=long_names,
        lat=lat,
        lon=lon,
        dates=tuple(parse_date(path) for path in files),
        fingerprint="-".join(fingerprint_items),
    )
    return _CATALOG


def clean_variable(name: str | None, catalog: Catalog) -> str:
    if not name:
        return "PM2.5" if "PM2.5" in catalog.variables else catalog.variables[0]
    decoded = unquote(str(name)).strip()
    if decoded in catalog.variables:
        return decoded
    lower = decoded.lower()
    alias = VARIABLE_ALIAS.get(lower)
    if alias in catalog.variables:
        return alias
    for key, value in VARIABLE_ALIAS.items():
        if key in lower and value in catalog.variables:
            return value
    for variable in catalog.variables:
        if variable.lower() == lower or lower in variable.lower():
            return variable
    return "PM2.5" if "PM2.5" in catalog.variables else catalog.variables[0]


def region_mask(region: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    bounds = REGION_BOUNDS.get(region)
    if bounds is None:
        return np.ones((lat.size, lon.size), dtype=bool)
    lon_min, lon_max, lat_min, lat_max = bounds
    return ((lat >= lat_min) & (lat <= lat_max))[:, None] & ((lon >= lon_min) & (lon <= lon_max))[None, :]


def cache_path(kind: str, *parts: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    safe_parts = [re.sub(r"[^A-Za-z0-9_.-]+", "_", part) for part in parts]
    return CACHE_DIR / f"{kind}_{'_'.join(safe_parts)}.json.gz"


def read_cache(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("_fingerprint") == fingerprint:
            return payload.get("data")
    except Exception:
        return None
    return None


def write_cache(path: Path, fingerprint: str, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        json.dump({"_fingerprint": fingerprint, "data": json_ready(data)}, handle, ensure_ascii=False)
    tmp.replace(path)


def load_variable_stack(variable: str, catalog: Catalog) -> np.ndarray:
    stack = np.empty((len(catalog.files), catalog.lat.size, catalog.lon.size), dtype=np.float32)
    for idx, path in enumerate(catalog.files):
        with xr.open_dataset(path, decode_times=False) as ds:
            stack[idx] = np.asarray(ds[variable].values, dtype=np.float32)
    return stack


def downsample_grid(values: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> dict[str, Any]:
    lat_step = max(1, int(math.ceil(values.shape[0] / MAX_GRID_POINTS)))
    lon_step = max(1, int(math.ceil(values.shape[1] / MAX_GRID_POINTS)))
    z = values[::lat_step, ::lon_step]
    return {
        "lat": [safe_float(v, 4) for v in lat[::lat_step]],
        "lon": [safe_float(v, 4) for v in lon[::lon_step]],
        "z": [[safe_float(v, 5) for v in row] for row in z],
    }


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None, "p05": None, "p50": None, "p95": None}
    return {
        "min": safe_float(np.nanmin(finite)),
        "max": safe_float(np.nanmax(finite)),
        "mean": safe_float(np.nanmean(finite)),
        "std": safe_float(np.nanstd(finite)),
        "p05": safe_float(np.nanpercentile(finite, 5)),
        "p50": safe_float(np.nanpercentile(finite, 50)),
        "p95": safe_float(np.nanpercentile(finite, 95)),
    }


def safe_nanmean(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return None
    return safe_float(float(np.mean(finite)))


def safe_nanstd(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return None
    return safe_float(float(np.std(finite)))


def nanmean_axis0(stack: np.ndarray) -> np.ndarray:
    valid = np.isfinite(stack)
    counts = valid.sum(axis=0)
    total = np.where(valid, stack, 0).sum(axis=0)
    return np.divide(total, counts, out=np.full(stack.shape[1:], np.nan, dtype=np.float32), where=counts > 0)


def nanstd_axis0(stack: np.ndarray, mean: np.ndarray) -> np.ndarray:
    valid = np.isfinite(stack)
    counts = valid.sum(axis=0)
    squared = np.where(valid, (stack - mean[None, :, :]) ** 2, 0).sum(axis=0)
    variance = np.divide(squared, counts, out=np.full(stack.shape[1:], np.nan, dtype=np.float32), where=counts > 0)
    return np.sqrt(variance)


def compute_trend_map(stack: np.ndarray) -> np.ndarray:
    x = np.arange(stack.shape[0], dtype=np.float32)
    x = x - x.mean()
    denominator = float(np.sum(x * x)) or 1.0
    mean = nanmean_axis0(stack)
    centered = np.where(np.isfinite(stack), stack - mean[None, :, :], 0)
    return np.tensordot(x, centered, axes=(0, 0)) / denominator


def connected_hotspots(mean_map: np.ndarray, lat: np.ndarray, lon: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    valid = np.isfinite(mean_map) & mask
    if not valid.any():
        return np.zeros_like(mean_map, dtype=float), []
    threshold = float(np.nanpercentile(mean_map[valid], 90))
    hotspot = (mean_map >= threshold) & valid
    labels, count = ndimage.label(hotspot)
    objects = ndimage.find_objects(labels)
    components: list[dict[str, Any]] = []
    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        component_mask = labels[slc] == label_id
        size = int(np.count_nonzero(component_mask))
        if size == 0:
            continue
        local_values = np.where(component_mask, mean_map[slc], np.nan)
        local_lat = lat[slc[0]]
        local_lon = lon[slc[1]]
        yy, xx = np.where(component_mask)
        components.append(
            {
                "id": label_id,
                "cells": size,
                "peak": safe_float(np.nanmax(local_values)),
                "mean": safe_float(np.nanmean(local_values)),
                "lat": safe_float(np.mean(local_lat[yy]), 4),
                "lon": safe_float(np.mean(local_lon[xx]), 4),
                "bbox": [
                    safe_float(local_lon.min(), 4),
                    safe_float(local_lat.min(), 4),
                    safe_float(local_lon.max(), 4),
                    safe_float(local_lat.max(), 4),
                ],
            }
        )
    components.sort(key=lambda item: (item["cells"], item["peak"] or -1), reverse=True)
    return hotspot.astype(float), components[:12]


def normalize01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo = float(np.nanpercentile(finite, 5))
    hi = float(np.nanpercentile(finite, 95))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


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

DEFAULT_SEMANTIC_THRESHOLDS = {
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


def clamp01(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


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
    directions = [
        (22.5, "E"),
        (67.5, "NE"),
        (112.5, "N"),
        (157.5, "NW"),
        (202.5, "W"),
        (247.5, "SW"),
        (292.5, "S"),
        (337.5, "SE"),
        (360.0, "E"),
    ]
    for upper, label in directions:
        if angle < upper:
            return label
    return "E"


def connected_component_stats(binary: np.ndarray) -> tuple[int, int]:
    if binary.size == 0 or not np.any(binary):
        return 0, 0
    labels, count = ndimage.label(binary)
    if count == 0:
        return 0, 0
    sizes = ndimage.sum(binary, labels, index=np.arange(1, count + 1))
    largest = int(np.max(sizes)) if len(sizes) else 0
    return int(count), largest


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
        left = centered[:-dy or None, :-dx or None] if dy or dx else centered
        right = centered[dy:, dx:]
        valid = np.isfinite(left) & np.isfinite(right)
        if np.count_nonzero(valid):
            numerator += float(np.nansum(left[valid] * right[valid])) * 2.0
            weights += int(np.count_nonzero(valid)) * 2
    if weights == 0:
        return None
    return safe_float((np.count_nonzero(finite) / weights) * (numerator / denominator), 4)


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
    evidence = [
        f"缺失率为 {round((features.get('missing_ratio') or 0) * 100, 1)}%",
        f"平均梯度强度归一化值为 {round(features.get('gradient_mean_norm') or 0, 3)}",
    ]
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
    elif label == "M":
        evidence = [
            "多个模式得分接近，单一标签区分度不足",
            f"高值占比 {round((features.get('hotspot_ratio') or 0) * 100, 1)}%，低值占比 {round((features.get('lowspot_ratio') or 0) * 100, 1)}%",
            f"边界强度归一化值为 {round(features.get('edge_strength_norm') or 0, 3)}",
        ]
    if (features.get("missing_ratio") or 0) <= 0.05:
        evidence.append("缺失率较低，数据质量较好")
    return evidence


def semantic_explanation(label: str, evidence: list[str]) -> str:
    label_name = GRID_PATTERN_LABELS[label]["name"]
    lead = {
        "H": "该 block 被识别为高值热点型，说明内部存在相对集中的局部高值中心。",
        "L": "该 block 被识别为低值冷点型，说明内部存在相对集中的局部低值中心。",
        "G": "该 block 被识别为梯度过渡型，说明数值沿某个方向呈连续变化。",
        "B": "该 block 被识别为边界突变型，说明内部或边缘存在明显突变带。",
        "D": "该 block 被识别为扩散分散型，说明异常斑块较多且不集中于单一中心。",
        "U": "该 block 被识别为均匀稳定型，说明内部空间变化较弱。",
        "M": "该 block 被识别为混合复杂型，说明多个空间结构同时存在且得分接近。",
        "N": "该 block 被识别为噪声不确定型，说明当前数据质量或稳定性不足以支持强判断。",
    }.get(label, f"该 block 被识别为{label_name}。")
    return f"{lead} 主要证据包括：{'；'.join(evidence[:3])}。"


def build_block_semantics(
    mean_map: np.ndarray,
    latest: np.ndarray,
    trend_map: np.ndarray,
    anomaly_map: np.ndarray,
    hotspot_mask: np.ndarray,
    stack_region: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
    *,
    rows: int = 9,
    cols: int = 14,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = {**DEFAULT_SEMANTIC_THRESHOLDS, **(thresholds or {})}
    lat_edges = np.linspace(0, len(lat), rows + 1, dtype=int)
    lon_edges = np.linspace(0, len(lon), cols + 1, dtype=int)
    analysis_map = np.asarray(latest, dtype=float)
    fallback_mean = float(np.nanmean(analysis_map[np.isfinite(analysis_map)])) if np.isfinite(analysis_map).any() else 0.0
    filled_map = np.nan_to_num(analysis_map, nan=fallback_mean)
    gy, gx = np.gradient(filled_map)
    gradient = np.sqrt(gx * gx + gy * gy)
    valid_global = np.isfinite(analysis_map) & mask
    global_values = analysis_map[valid_global]
    if global_values.size == 0:
        global_values = mean_map[np.isfinite(mean_map) & mask]
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
    valid_grad = gradient[valid_global]
    grad_p95 = float(np.nanpercentile(valid_grad, 95)) if valid_grad.size else 1.0
    grad_p95 = max(grad_p95, 1e-9)

    raw_blocks: list[dict[str, Any]] = []
    for row in range(rows):
        for col in range(cols):
            y0, y1 = int(lat_edges[row]), int(lat_edges[row + 1])
            x0, x1 = int(lon_edges[col]), int(lon_edges[col + 1])
            block_mask = mask[y0:y1, x0:x1]
            if not block_mask.any():
                continue
            local = analysis_map[y0:y1, x0:x1]
            local_valid_mask = block_mask & np.isfinite(local)
            latest_values = local[local_valid_mask]
            valid_cells = int(np.count_nonzero(local_valid_mask))
            mask_cells = int(np.count_nonzero(block_mask))
            if valid_cells == 0:
                continue
            mean_values = mean_map[y0:y1, x0:x1][local_valid_mask]
            trend_values = trend_map[y0:y1, x0:x1][block_mask]
            anomaly_values = anomaly_map[y0:y1, x0:x1][block_mask]
            block_gradient = gradient[y0:y1, x0:x1]
            gradient_values = block_gradient[local_valid_mask]
            block_gx = gx[y0:y1, x0:x1]
            block_gy = gy[y0:y1, x0:x1]
            high_binary = (local >= high_threshold) & local_valid_mask
            low_binary = (local <= low_threshold) & local_valid_mask
            high_components, high_largest = connected_component_stats(high_binary)
            low_components, low_largest = connected_component_stats(low_binary)
            largest_component = max(high_largest, low_largest)
            outliers = np.abs((latest_values - median) / outlier_scale) > 3.5
            block_flat = stack_region[:, y0:y1, x0:x1].reshape(stack_region.shape[0], -1)
            block_series = np.array([safe_nanmean(row) for row in block_flat], dtype=float)
            valid_series = block_series[np.isfinite(block_series)]
            volatility = float(np.std(valid_series)) if valid_series.size else np.nan
            q25 = safe_float(np.nanpercentile(latest_values, 25))
            q50 = safe_float(np.nanpercentile(latest_values, 50))
            q75 = safe_float(np.nanpercentile(latest_values, 75))
            value_min = float(np.nanmin(latest_values))
            value_max = float(np.nanmax(latest_values))
            value_mean = float(np.nanmean(latest_values))
            value_std = float(np.nanstd(latest_values))
            value_range = value_max - value_min
            gradient_mean = float(np.nanmean(gradient_values)) if gradient_values.size else 0.0
            gradient_std = float(np.nanstd(gradient_values)) if gradient_values.size else 0.0
            edge_strength = float(np.nanpercentile(gradient_values, 90)) if gradient_values.size else 0.0
            hotspot_ratio = float(np.count_nonzero(high_binary) / max(valid_cells, 1))
            lowspot_ratio = float(np.count_nonzero(low_binary) / max(valid_cells, 1))
            largest_component_ratio = float(largest_component / max(valid_cells, 1))
            missing_ratio = float(1.0 - valid_cells / max(mask_cells, 1))
            outlier_ratio = float(np.count_nonzero(outliers) / max(valid_cells, 1))
            features = {
                "mean": safe_float(value_mean),
                "std": safe_float(value_std),
                "min": safe_float(value_min),
                "max": safe_float(value_max),
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "value_range": safe_float(value_range),
                "gradient_mean": safe_float(gradient_mean),
                "gradient_std": safe_float(gradient_std),
                "dominant_gradient_direction": dominant_gradient_direction(block_gx[local_valid_mask], block_gy[local_valid_mask]),
                "edge_strength": safe_float(edge_strength),
                "hotspot_ratio": safe_float(hotspot_ratio, 4),
                "lowspot_ratio": safe_float(lowspot_ratio, 4),
                "connected_components_high": high_components,
                "connected_components_low": low_components,
                "largest_component_ratio": safe_float(largest_component_ratio, 4),
                "moran_i": simple_moran_i(np.where(block_mask, local, np.nan)),
                "missing_ratio": safe_float(missing_ratio, 4),
                "outlier_ratio": safe_float(outlier_ratio, 4),
                "std_norm": safe_float(value_std / global_range, 4),
                "gradient_mean_norm": safe_float(gradient_mean / grad_p95, 4),
                "edge_strength_norm": safe_float(edge_strength / grad_p95, 4),
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
            raw_blocks.append(
                {
                    "block_id": f"B_{row:03d}_{col:03d}",
                    "id": f"B_{row:03d}_{col:03d}",
                    "row": row,
                    "col": col,
                    "bbox": {
                        "lon_min": safe_float(lon[x0], 4),
                        "lon_max": safe_float(lon[min(x1 - 1, len(lon) - 1)], 4),
                        "lat_min": safe_float(lat[y0], 4),
                        "lat_max": safe_float(lat[min(y1 - 1, len(lat) - 1)], 4),
                    },
                    "center": {
                        "lat": safe_float(float(np.mean(lat[y0:y1])), 4),
                        "lon": safe_float(float(np.mean(lon[x0:x1])), 4),
                    },
                    "primary_label": primary_label,
                    "primary_label_name": GRID_PATTERN_LABELS[primary_label]["name"],
                    "secondary_labels": secondary_labels,
                    "confidence": safe_float(confidence, 4),
                    "scores": scores,
                    "features": features,
                    "evidence": evidence,
                    "llm_explanation": semantic_explanation(primary_label, evidence),
                    "mean": safe_nanmean(mean_values),
                    "latest": safe_nanmean(latest_values),
                    "trend": safe_nanmean(trend_values),
                    "anomaly": safe_nanmean(np.abs(anomaly_values)),
                    "gradient": safe_float(gradient_mean),
                    "hotspot_ratio": safe_float(hotspot_ratio, 4),
                    "volatility": safe_float(volatility),
                    "finite_ratio": safe_float(valid_cells / max(mask_cells, 1), 4),
                    "saliency": safe_float(max(scores.get("H", 0), scores.get("L", 0), scores.get("G", 0), scores.get("B", 0), scores.get("D", 0)), 4),
                    "pattern_type": primary_label,
                    "trend_label": "rising" if safe_nanmean(trend_values) and (safe_nanmean(trend_values) or 0) > 1e-5 else "falling" if safe_nanmean(trend_values) and (safe_nanmean(trend_values) or 0) < -1e-5 else "stable",
                    "uncertainty": "high" if primary_label == "N" else "medium" if scores.get("N", 0) > 0.45 else "low",
                }
            )

    matrix = np.full((rows, cols), np.nan, dtype=float)
    label_matrix: list[list[str | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    confidence_matrix = np.full((rows, cols), np.nan, dtype=float)
    blocks = []
    for block in raw_blocks:
        block["llm_tokens"] = [
            f"label:{block['primary_label']}",
            f"confidence:{block['confidence']}",
            f"trend:{block['trend_label']}",
            f"risk:{block['uncertainty']}",
        ]
        matrix[block["row"], block["col"]] = block["saliency"]
        confidence_matrix[block["row"], block["col"]] = block["confidence"]
        label_matrix[block["row"]][block["col"]] = block["primary_label"]
        blocks.append(block)

    blocks.sort(key=lambda item: item["saliency"] or 0, reverse=True)
    label_counts = {key: 0 for key in GRID_PATTERN_LABELS}
    for block in blocks:
        label_counts[block["primary_label"]] = label_counts.get(block["primary_label"], 0) + 1
    return {
        "grid": {"rows": rows, "cols": cols},
        "matrix": [[safe_float(v, 4) for v in row] for row in matrix],
        "label_matrix": label_matrix,
        "confidence_matrix": [[safe_float(v, 4) for v in row] for row in confidence_matrix],
        "blocks": blocks,
        "top_blocks": blocks[:12],
        "label_distribution": label_counts,
        "labels": GRID_PATTERN_LABELS,
        "thresholds": thresholds,
        "source_note": {
            "label": "Pattern Label generated by Block Semantic Extractor.",
            "explanation": "Explanation generated by LLM Assistant.",
            "combined": "Algorithmic Label + LLM Explanation.",
        },
        "semantic_schema": {
            "labels": GRID_PATTERN_LABELS,
            "classifier": "deterministic rule baseline",
            "features": [
                "mean",
                "std",
                "min",
                "max",
                "q25",
                "q50",
                "q75",
                "value_range",
                "gradient_mean",
                "gradient_std",
                "dominant_gradient_direction",
                "edge_strength",
                "hotspot_ratio",
                "lowspot_ratio",
                "connected_components_high",
                "connected_components_low",
                "largest_component_ratio",
                "moran_i",
                "missing_ratio",
                "outlier_ratio",
            ],
            "tokens": ["primary_label", "secondary_labels", "confidence", "evidence", "uncertainty"],
        },
    }


def similar_time_slices(
    stack_region: np.ndarray,
    dates: tuple[str, ...],
    *,
    query_index: int | None = None,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    query_index = len(dates) - 1 if query_index is None else max(0, min(int(query_index), len(dates) - 1))
    target = stack_region[query_index].reshape(-1)
    valid_target = np.isfinite(target)
    results = []
    for idx in range(stack_region.shape[0]):
        if idx == query_index:
            continue
        candidate = stack_region[idx].reshape(-1)
        valid = valid_target & np.isfinite(candidate)
        if np.count_nonzero(valid) < 200:
            continue
        x = target[valid]
        y = candidate[valid]
        if float(np.std(x)) == 0 or float(np.std(y)) == 0:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        rmse = float(np.sqrt(np.mean((x - y) ** 2)))
        results.append({"date": dates[idx], "similarity": safe_float(corr, 4), "rmse": safe_float(rmse, 4)})
    results.sort(key=lambda item: item["similarity"] or -1, reverse=True)
    return results[:top_k]


def daily_variable_matrix(catalog: Catalog) -> dict[str, Any]:
    path = cache_path("daily", catalog.fingerprint)
    cached = read_cache(path, catalog.fingerprint)
    if cached:
        return cached

    means: dict[str, list[float | None]] = {name: [] for name in catalog.variables}
    p95s: dict[str, list[float | None]] = {name: [] for name in catalog.variables}
    for file_path in catalog.files:
        with xr.open_dataset(file_path, decode_times=False) as ds:
            for name in catalog.variables:
                arr = np.asarray(ds[name].values, dtype=np.float32)
                finite = arr[np.isfinite(arr)]
                means[name].append(safe_float(np.mean(finite)) if finite.size else None)
                p95s[name].append(safe_float(np.percentile(finite, 95)) if finite.size else None)

    payload = {"dates": list(catalog.dates), "mean": means, "p95": p95s}
    write_cache(path, catalog.fingerprint, payload)
    return payload


def pearson(x: list[float | None], y: list[float | None]) -> float | None:
    xa = np.asarray([np.nan if v is None else v for v in x], dtype=float)
    ya = np.asarray([np.nan if v is None else v for v in y], dtype=float)
    valid = np.isfinite(xa) & np.isfinite(ya)
    if np.count_nonzero(valid) < 3:
        return None
    xs = xa[valid]
    ys = ya[valid]
    if float(np.std(xs)) == 0 or float(np.std(ys)) == 0:
        return None
    return safe_float(np.corrcoef(xs, ys)[0, 1], 4)


def task_type_from_query(query: str) -> str:
    text = query.lower()
    if any(token in text for token in ["趋势", "变化", "时间", "trend", "evolution"]):
        return "spatiotemporal_trend"
    if any(token in text for token in ["异常", "极端", "污染事件", "anomaly", "outlier"]):
        return "anomaly_detection"
    if any(token in text for token in ["相关", "影响", "驱动", "归因", "correlation", "attribution"]):
        return "attribution"
    if any(token in text for token in ["热点", "高值", "hotspot"]):
        return "hotspot_discovery"
    return "distribution_diagnosis"


def visualization_strategy_from_semantics(block_semantics: dict[str, Any]) -> dict[str, Any]:
    distribution = block_semantics.get("label_distribution") or {}
    total = max(1, sum(int(value or 0) for value in distribution.values()))
    ratios = {label: (int(count or 0) / total) for label, count in distribution.items()}
    recommendations: list[dict[str, Any]] = []
    rules = [
        ("H", "热点地图与 Top-K 高值区域列表", ["热点地图", "Top-K 高值区域列表", "局部放大视图"]),
        ("L", "低值冷点地图与低值区摘要", ["冷点地图", "低值区域排名", "局部对比视图"]),
        ("G", "梯度方向图与剖面分析", ["梯度方向图", "剖面图", "方向性箭头叠加"]),
        ("B", "边界突变图与边界线高亮", ["边界突变图", "差值图", "边界线高亮"]),
        ("D", "连通域图与斑块统计", ["连通域图", "斑块统计图", "分散度分析"]),
        ("N", "不确定性掩膜与质量提示", ["不确定性掩膜", "数据质量提示", "谨慎解释"]),
        ("U", "背景场总览与稳定区标注", ["均匀背景图", "稳定区摘要", "低变化区域标注"]),
        ("M", "多图层联动与复杂区钻取", ["多标签叠加", "复杂区钻取", "证据面板"]),
    ]
    for label, summary, views in rules:
        ratio = ratios.get(label, 0.0)
        if ratio >= 0.12 or (label in {"H", "G", "B", "D", "N"} and ratio >= 0.08):
            recommendations.append(
                {
                    "trigger_label": label,
                    "label_name": GRID_PATTERN_LABELS[label]["name"],
                    "ratio": safe_float(ratio, 4),
                    "summary": summary,
                    "recommended_views": views,
                    "constraint": "LLM may explain or prioritize this recommendation, but chart rendering is executed by the tool layer.",
                }
            )
    recommendations.sort(key=lambda item: item["ratio"], reverse=True)
    return {
        "source": "Rule-constrained LLM strategy interface",
        "label_distribution": distribution,
        "label_ratios": {key: safe_float(value, 4) for key, value in ratios.items()},
        "recommendations": recommendations[:6],
        "responsibility": {
            "label_generation": "Block Semantic Extractor algorithm",
            "strategy_text": "LLM Assistant constrained by semantic-label rule library",
            "chart_execution": "Tool execution layer",
        },
    }


def build_llm_package(query: str, variable: str, region: str, analysis: dict[str, Any]) -> dict[str, Any]:
    task_type = task_type_from_query(query)
    top_corr = analysis["correlations"]["daily"][:3]
    hotspot_count = len(analysis["hotspots"])
    stats = analysis["statistics"]["overall"]
    latest = analysis["statistics"]["latest"]
    trend = analysis["statistics"]["temporal_trend"]
    unit = analysis["selection"]["unit"] or ""
    unit_text = f" {unit}" if unit else ""
    slope_value = trend.get("slope") or 0
    delta_value = trend.get("delta") or 0
    direction = "上升" if slope_value > 0 and delta_value >= 0 else "下降" if slope_value < 0 and delta_value <= 0 else "波动"
    if abs(slope_value) < 1e-5:
        direction = "基本稳定"

    corr_text = "、".join(
        f"{item['variable']}({item['correlation']:+.2f})"
        for item in top_corr
        if item.get("correlation") is not None
    ) or "暂无显著日均相关项"
    block_summary = analysis.get("block_semantics", {})
    top_blocks = block_summary.get("top_blocks") or []
    block_text = "、".join(
        f"{item['id']}[{item.get('primary_label')}/{item.get('trend_label')}]"
        for item in top_blocks[:3]
    ) or "暂无高显著性语义块"
    semantic_strategy = visualization_strategy_from_semantics(block_summary)

    return {
        "query": query,
        "task_graph": {
            "task_type": task_type,
            "region": REGION_LABELS.get(region, region),
            "variables": [variable],
            "time_range": f"{analysis['catalog']['date_start']} 至 {analysis['catalog']['date_end']}",
            "analysis_goals": [
                "识别时空分布格局",
                "定位高值热点连通域与块级语义模式",
                "评估时序趋势和变量耦合",
                "输出可追溯解释与风险提示",
            ],
        },
        "visualization_strategy": {
            "layout": "semantic block map + block evidence panel + temporal retrieval + LLM-native explanation",
            "views": [
                "Heatmap",
                "Semantic Blocks",
                "日均与 P95 时间序列",
                "相似历史格局检索",
                "变量相关网络",
                "Trace & Provenance",
            ],
            "encodings": {
                "color": "continuous grid values plus categorical H/L/G/B/D/U/M/N block labels",
                "block": "primary_label, secondary_labels, confidence, evidence, uncertainty",
                "space": "lat/lon gridded heatmap",
                "time": "daily aggregation over 152 NetCDF files",
            },
            "semantic_strategy": semantic_strategy,
            "responsibility": semantic_strategy["responsibility"],
        },
        "narrative": (
            f"{REGION_LABELS.get(region, region)}在 {analysis['catalog']['date_start']} 至 "
            f"{analysis['catalog']['date_end']} 期间的 {variable} 均值为 {stats.get('mean')}{unit_text}，"
            f"最新日均值为 {latest.get('mean')}{unit_text}。90 分位阈值提取到 {hotspot_count} 个主要热点连通域，"
            f"全域日均趋势表现为{direction}，日变化斜率约 {trend.get('slope')}{unit_text}/day。"
            f"块级语义摘要识别出的重点块为：{block_text}。"
            f"与目标变量日均序列耦合最强的变量包括：{corr_text}。"
        ),
        "uncertainty_risks": [
            "当前默认使用本地规则和确定性统计生成解释；若显式配置远程 LLM API，可进一步生成更细粒度的论文式叙述。",
            "数据已被统一到规则网格，极端值和插值区域需要结合原始观测站点或遥感源进一步审计。",
            "变量相关性不等同于因果归因，污染过程仍需排放、边界层、传输路径等外部证据支撑。",
        ],
        "follow_up_questions": [
            f"{variable} 高值区是否与温湿度、风速或降水存在滞后关系？",
            "是否需要对京津冀、长三角、珠三角进行区域对比？",
            "是否生成可投稿论文中的方法流程图和高分辨率图件？",
        ],
    }


def compact_features_for_planner(analysis: dict[str, Any]) -> dict[str, Any]:
    selection = analysis.get("selection", {})
    stats = analysis.get("statistics", {})
    correlations = analysis.get("correlations", {}).get("daily", [])
    blocks = analysis.get("block_semantics", {}).get("top_blocks", [])
    return {
        "global_statistics": {
            "primary_variable": selection.get("variable"),
            "variables": {
                item.get("name"): {
                    "mean": item.get("mean"),
                    "trend": item.get("trend"),
                    "unit": item.get("unit"),
                }
                for item in analysis.get("variables", [])
                if item.get("name")
            },
            "correlation": {
                item.get("variable"): item.get("correlation")
                for item in correlations
                if item.get("variable") and item.get("correlation") is not None
            },
        },
        "spatial_features": {
            "primary_variable": selection.get("variable"),
            "hotspot_count": len(analysis.get("hotspots", [])),
            "top_hotspots": analysis.get("hotspots", [])[:3],
            "semantic_blocks": blocks[:5],
        },
        "temporal_features": {
            "primary_variable": selection.get("variable"),
            "trend": stats.get("temporal_trend", {}).get("direction"),
            "change_points": analysis.get("temporal", {}).get("anomaly_days", [])[:3],
        },
        "semantic_summary": {
            "primary_variable": selection.get("variable"),
            "region": selection.get("region_label"),
            "selected_date": selection.get("selected_date") or selection.get("latest_date"),
        },
        "metadata": {
            "tensor_shape": [
                len(analysis.get("temporal", {}).get("dates", [])),
                len(analysis.get("variables", [])),
                *(analysis.get("catalog", {}).get("shape") or []),
            ],
            "source_summary": analysis.get("provenance", {}),
        },
    }


def enrich_llm_package(query: str, base: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("GRIDVIS_DISABLE_LLM_PLANNER") == "1":
        return base
    try:
        from agents import run_llm_agents

        agent_result = run_llm_agents(query, analysis)
        if isinstance(agent_result.get("task_graph"), dict):
            base["task_graph"].update(agent_result["task_graph"])
        if isinstance(agent_result.get("visualization_strategy"), dict):
            base["visualization_strategy"].update(agent_result["visualization_strategy"])
        if agent_result.get("narrative"):
            base["narrative"] = agent_result["narrative"]
        if agent_result.get("uncertainty_risks"):
            base["uncertainty_risks"] = agent_result["uncertainty_risks"]
        if agent_result.get("follow_up_questions"):
            base["follow_up_questions"] = agent_result["follow_up_questions"]
        base["agent_outputs"] = agent_result.get("agent_outputs", {})
        base["llm_metadata"] = agent_result.get("llm_metadata", {})
        return base
    except Exception as exc:
        base["agent_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from llm_planner import plan_visualization

        use_remote_llm = os.getenv("GRIDVIS_USE_REMOTE_LLM") == "1"
        plan = plan_visualization(query, compact_features_for_planner(analysis), use_llm=use_remote_llm)
    except Exception as exc:
        base["llm_metadata"] = {
            "mode": "local_package",
            "planner_error": f"{type(exc).__name__}: {exc}",
        }
        return base

    if isinstance(plan.get("task_graph"), dict):
        base["task_graph"].update(plan["task_graph"])
    strategy = plan.get("visualization_strategy")
    if isinstance(strategy, dict):
        base["visualization_strategy"]["planner"] = strategy
        chart_type = strategy.get("chart_type")
        if chart_type and not base["visualization_strategy"].get("views"):
            base["visualization_strategy"]["views"] = [chart_type]
    if plan.get("semantic_response"):
        base["narrative"] = plan["semantic_response"]
    if plan.get("uncertainty_risks"):
        base["uncertainty_risks"] = plan["uncertainty_risks"]
    base["llm_metadata"] = plan.get("llm_metadata", {})
    if base.get("agent_error"):
        base["llm_metadata"]["agent_error"] = base["agent_error"]
    return base


def build_module_views(analysis: dict[str, Any]) -> dict[str, Any]:
    """Build module-level visual contracts for the frontend.

    These objects describe the processing states of the five GridVis-LLM
    modules. They intentionally avoid raw-array dumps and expose compact
    visual primitives: nodes, edges, glyphs, stage scores, and provenance links.
    """

    blocks = analysis.get("block_semantics", {}).get("blocks", [])
    top_blocks = analysis.get("block_semantics", {}).get("top_blocks", [])
    patterns: dict[str, int] = {}
    uncertainties: dict[str, int] = {}
    for block in blocks:
        patterns[block.get("pattern_type", "unknown")] = patterns.get(block.get("pattern_type", "unknown"), 0) + 1
        uncertainties[block.get("uncertainty", "unknown")] = uncertainties.get(block.get("uncertainty", "unknown"), 0) + 1

    correlations = analysis.get("correlations", {}).get("daily", [])
    retrieval = analysis.get("retrieval", {}).get("similar_time_slices", [])
    llm = analysis.get("llm", {})
    stats = analysis.get("statistics", {})
    selection = analysis.get("selection", {})
    catalog = analysis.get("catalog", {})

    policy_weight_map = {
        "semantic_field": min(1.0, 0.55 + 0.04 * len(top_blocks)),
        "block_matrix": min(1.0, 0.45 + 0.02 * len(blocks) / 10),
        "temporal_retrieval": min(1.0, 0.35 + 0.07 * len(retrieval)),
        "coupling_policy": min(1.0, 0.35 + 0.05 * len(correlations)),
        "provenance": 0.86,
    }
    policy_nodes = [
        {
            "id": key,
            "label": label,
            "weight": safe_float(policy_weight_map[key], 4),
            "role": role,
        }
        for key, label, role in [
            ("semantic_field", "Semantic Field", "spatial block reasoning"),
            ("block_matrix", "Block Matrix", "multi-scale compression"),
            ("temporal_retrieval", "Similar Episodes", "historical pattern retrieval"),
            ("coupling_policy", "Coupling Policy", "multi-variable evidence"),
            ("provenance", "LLM-native Figure", "traceable output"),
        ]
    ]

    execution_steps = [
        {
            "id": "scan",
            "label": "scan files",
            "status": "done",
            "metric": f"{catalog.get('file_count')} slices",
            "tool": "xarray",
        },
        {
            "id": "tensor",
            "label": "standardize tensor",
            "status": "done",
            "metric": "time x var x lat x lon",
            "tool": "GridTensor",
        },
        {
            "id": "blocks",
            "label": "semantic blocks",
            "status": "done",
            "metric": f"{len(blocks)} blocks",
            "tool": "numpy/scipy",
        },
        {
            "id": "retrieve",
            "label": "retrieve episodes",
            "status": "done",
            "metric": f"{len(retrieval)} matches",
            "tool": "Pearson field similarity",
        },
        {
            "id": "bind",
            "label": "bind figure",
            "status": "done",
            "metric": "5 evidence views",
            "tool": "LLM-native contract",
        },
    ]

    provenance_figures = [
        {
            "id": "fig-semantic-field",
            "figure": "Semantic Block Field",
            "data_slice": f"{selection.get('variable')} / {selection.get('region_label')}",
            "evidence": [block.get("id") for block in top_blocks[:4]],
            "parameters": ["layer", "block saliency", "pattern outlines"],
        },
        {
            "id": "fig-retrieval",
            "figure": "Temporal Retrieval",
            "data_slice": f"{catalog.get('date_start')} - {catalog.get('date_end')}",
            "evidence": [item.get("date") for item in retrieval[:4]],
            "parameters": ["daily mean", "p95", "spatial similarity"],
        },
        {
            "id": "fig-policy",
            "figure": "Policy Attention",
            "data_slice": llm.get("task_graph", {}).get("task_type"),
            "evidence": [node["id"] for node in policy_nodes[:5]],
            "parameters": ["intent", "pattern profile", "uncertainty"],
        },
    ]

    return {
        "adapter": {
            "title": "Grid Semantic Adapter",
            "input_ports": [
                {"id": "files", "label": "NetCDF", "value": catalog.get("file_count")},
                {"id": "vars", "label": "Variables", "value": len(analysis.get("variables", []))},
                {"id": "time", "label": "Time", "value": f"{catalog.get('date_start')} -> {catalog.get('date_end')}"},
                {"id": "mask", "label": "Mask", "value": stats.get("quality", {}).get("finite_ratio")},
            ],
            "output": {
                "label": "GridTensor",
                "shape": catalog.get("shape"),
                "variable": selection.get("variable"),
                "unit": selection.get("unit"),
            },
            "quality": stats.get("quality", {}),
        },
        "representation": {
            "title": "Block Semantic Extractor",
            "grid": analysis.get("block_semantics", {}).get("grid"),
            "pattern_distribution": [
                {"pattern": key, "count": value}
                for key, value in sorted(patterns.items(), key=lambda item: item[1], reverse=True)
            ],
            "uncertainty_distribution": [
                {"level": key, "count": value}
                for key, value in sorted(uncertainties.items(), key=lambda item: item[0])
            ],
            "top_blocks": top_blocks[:10],
            "schema": analysis.get("block_semantics", {}).get("semantic_schema", {}),
        },
        "llm_core": {
            "title": "Visualization Policy Engine",
            "query": llm.get("query"),
            "task_graph": llm.get("task_graph", {}),
            "policy_nodes": policy_nodes,
            "policy_edges": [
                {"source": "semantic_field", "target": "block_matrix", "weight": 0.72},
                {"source": "semantic_field", "target": "temporal_retrieval", "weight": 0.64},
                {"source": "block_matrix", "target": "provenance", "weight": 0.78},
                {"source": "coupling_policy", "target": "provenance", "weight": 0.58},
                {"source": "temporal_retrieval", "target": "provenance", "weight": 0.66},
            ],
            "selected_views": llm.get("visualization_strategy", {}).get("views", []),
        },
        "execution": {
            "title": "Tool Execution Graph",
            "steps": execution_steps,
            "runtime": analysis.get("runtime", {}),
            "artifacts": ["semantic field", "block matrix", "retrieval timeline", "coupling bars", "provenance cards"],
        },
        "output": {
            "title": "LLM-native Figure Binder",
            "figures": provenance_figures,
            "risks": llm.get("uncertainty_risks", []),
            "followups": llm.get("follow_up_questions", []),
        },
    }


def build_analysis(
    variable: str,
    region: str = "china",
    query: str | None = None,
    target_date: str | None = None,
) -> dict[str, Any]:
    catalog = get_catalog()
    variable = clean_variable(variable, catalog)
    region = region if region in REGION_BOUNDS else "china"
    selected_index, selected_date = select_date_index(catalog.dates, target_date or extract_query_date(query))
    cache = cache_path("analysis", ANALYSIS_CACHE_VERSION, catalog.fingerprint, variable, region, selected_date)
    cached = read_cache(cache, catalog.fingerprint)
    if cached:
        if query:
            cached["llm"] = enrich_llm_package(query, build_llm_package(query, variable, region, cached), cached)
        cached["module_views"] = build_module_views(cached)
        cached["runtime"] = {"cache": "hit", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        return cached

    started = time.time()
    mask = region_mask(region, catalog.lat, catalog.lon)
    stack = load_variable_stack(variable, catalog)
    stack_region = np.where(mask[None, :, :], stack, np.nan)
    latest = stack_region[selected_index]
    mean_map = nanmean_axis0(stack_region)
    std_map = nanstd_axis0(stack_region, mean_map)
    trend_map = compute_trend_map(stack_region)
    anomaly_map = np.divide(latest - mean_map, std_map, out=np.zeros_like(mean_map), where=std_map > 1e-9)
    hotspot_mask, hotspots = connected_hotspots(mean_map, catalog.lat, catalog.lon, mask)
    block_semantics = build_block_semantics(
        mean_map,
        latest,
        trend_map,
        anomaly_map,
        hotspot_mask,
        stack_region,
        catalog.lat,
        catalog.lon,
        mask,
    )
    similar_slices = similar_time_slices(stack_region, catalog.dates, query_index=selected_index)

    daily_mean = np.nanmean(stack_region, axis=(1, 2))
    daily_p95 = np.nanpercentile(stack_region.reshape(stack_region.shape[0], -1), 95, axis=1)
    daily_max = np.nanmax(stack_region, axis=(1, 2))
    threshold = np.nanpercentile(mean_map[np.isfinite(mean_map)], 90)
    hotspot_ratio = np.nanmean(stack_region >= threshold, axis=(1, 2))

    x = np.arange(len(daily_mean), dtype=float)
    slope = safe_float(np.polyfit(x, daily_mean, 1)[0]) if len(daily_mean) > 2 else None
    smoothed = ndimage.gaussian_filter1d(daily_mean.astype(float), sigma=2)
    residual = daily_mean - smoothed
    anomaly_days = np.argsort(np.abs(residual))[-8:][::-1]

    daily_matrix = daily_variable_matrix(catalog)
    correlations = []
    for name in catalog.variables:
        if name == variable:
            continue
        value = pearson(daily_matrix["mean"][variable], daily_matrix["mean"][name])
        correlations.append(
            {
                "variable": name,
                "label": catalog.variable_names.get(name) or name,
                "correlation": value,
                "abs": abs(value) if value is not None else -1,
            }
        )
    correlations.sort(key=lambda item: item["abs"], reverse=True)

    variable_cards = []
    for name in catalog.variables:
        series = daily_matrix["mean"].get(name, [])
        variable_cards.append(
            {
                "name": name,
                "label": catalog.variable_names.get(name) or name,
                "unit": catalog.variable_units.get(name),
                "mean": safe_float(np.nanmean([np.nan if v is None else v for v in series])),
                "trend": safe_float(np.polyfit(x, np.asarray([np.nan if v is None else v for v in series], dtype=float), 1)[0])
                if len(series) > 2
                else None,
                "selected": name == variable,
            }
        )

    payload = {
        "catalog": {
            "data_dir": str(DATA_DIR),
            "file_count": len(catalog.files),
            "date_start": catalog.dates[0],
            "date_end": catalog.dates[-1],
            "shape": [int(catalog.lat.size), int(catalog.lon.size)],
            "bounds": {
                "lat_min": safe_float(catalog.lat.min(), 4),
                "lat_max": safe_float(catalog.lat.max(), 4),
                "lon_min": safe_float(catalog.lon.min(), 4),
                "lon_max": safe_float(catalog.lon.max(), 4),
            },
        },
        "selection": {
            "variable": variable,
            "label": catalog.variable_names.get(variable) or variable,
            "unit": catalog.variable_units.get(variable),
            "region": region,
            "region_label": REGION_LABELS.get(region, region),
            "latest_date": selected_date,
            "selected_date": selected_date,
            "selected_index": selected_index,
        },
        "variables": variable_cards,
        "statistics": {
            "overall": finite_stats(stack_region),
            "latest": finite_stats(latest),
            "mean_map": finite_stats(mean_map),
            "temporal_trend": {
                "slope": slope,
                "direction": "increase" if (slope or 0) > 0 else "decrease" if (slope or 0) < 0 else "stable",
                "first_mean": safe_float(daily_mean[0]),
                "last_mean": safe_float(daily_mean[-1]),
                "delta": safe_float(daily_mean[-1] - daily_mean[0]),
            },
            "quality": {
                "finite_ratio": safe_float(np.isfinite(stack_region).mean(), 5),
                "valid_region_cells": int(np.count_nonzero(mask)),
                "total_cells": int(mask.size),
            },
        },
        "temporal": {
            "dates": list(catalog.dates),
            "mean": [safe_float(v) for v in daily_mean],
            "p95": [safe_float(v) for v in daily_p95],
            "max": [safe_float(v) for v in daily_max],
            "hotspot_ratio": [safe_float(v, 5) for v in hotspot_ratio],
            "anomaly_days": [
                {
                    "date": catalog.dates[int(idx)],
                    "mean": safe_float(daily_mean[int(idx)]),
                    "residual": safe_float(residual[int(idx)]),
                }
                for idx in anomaly_days
            ],
        },
        "maps": {
            "mean": downsample_grid(mean_map, catalog.lat, catalog.lon),
            "latest": downsample_grid(latest, catalog.lat, catalog.lon),
            "trend": downsample_grid(trend_map, catalog.lat, catalog.lon),
            "anomaly": downsample_grid(anomaly_map, catalog.lat, catalog.lon),
            "hotspot": downsample_grid(hotspot_mask, catalog.lat, catalog.lon),
        },
        "hotspots": hotspots,
        "block_semantics": block_semantics,
        "retrieval": {
            "query_slice": selected_date,
            "similar_time_slices": similar_slices,
            "method": "Pearson similarity between latest spatial field and historical daily fields",
        },
        "correlations": {
            "daily": [
                {k: v for k, v in item.items() if k != "abs"}
                for item in correlations
                if item.get("correlation") is not None
            ][:12],
        },
        "provenance": {
            "source_files": [str(catalog.files[0]), str(catalog.files[-1])],
            "source_file_count": len(catalog.files),
            "pipeline": [
                "NetCDF catalog scan",
                "GridTensor-compatible variable stack",
                "region masking",
                "daily aggregation",
                "hotspot connected components",
                "trend and anomaly fields",
                "grid block semantic summarization",
                "historical pattern retrieval",
                "LLM-ready evidence synthesis",
            ],
            "backend": "gridvis_server.py / Python stdlib HTTP API",
            "cache_file": str(cache),
            "compute_seconds": safe_float(time.time() - started, 3),
        },
    }
    effective_query = query or f"分析{variable}的时空分布、热点和趋势"
    payload["llm"] = enrich_llm_package(
        effective_query,
        build_llm_package(effective_query, variable, region, payload),
        payload,
    )
    payload["runtime"] = {"cache": "miss", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    payload["module_views"] = build_module_views(payload)
    write_cache(cache, catalog.fingerprint, payload)
    return payload


def build_catalog_payload() -> dict[str, Any]:
    catalog = get_catalog()
    return {
        "data_dir": str(DATA_DIR),
        "file_count": len(catalog.files),
        "date_start": catalog.dates[0],
        "date_end": catalog.dates[-1],
        "shape": [int(catalog.lat.size), int(catalog.lon.size)],
        "variables": [
            {
                "name": name,
                "label": catalog.variable_names.get(name) or name,
                "unit": catalog.variable_units.get(name),
            }
            for name in catalog.variables
        ],
        "regions": [{"id": key, "label": label} for key, label in REGION_LABELS.items()],
    }


class GridVisHandler(BaseHTTPRequestHandler):
    server_version = "GridVisLLM/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/frontend_index.html"}:
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", "/frontend/final/index.html")
                self.end_headers()
            elif parsed.path in {"/frontend/final/index.html", "/03_llm_core_frontend/index.html"}:
                self.send_file(FRONTEND_FILE, "text/html; charset=utf-8")
            elif parsed.path == "/api/health":
                self.send_json({"ok": True, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            elif parsed.path == "/api/catalog":
                self.send_json(build_catalog_payload())
            elif parsed.path == "/api/analysis":
                params = parse_qs(parsed.query)
                catalog = get_catalog()
                variable = clean_variable((params.get("variable") or [None])[0], catalog)
                region = (params.get("region") or ["china"])[0]
                query = (params.get("query") or [None])[0]
                target_date = (params.get("date") or [None])[0]
                self.send_json(build_analysis(variable, region=region, query=query, target_date=target_date))
            else:
                self.serve_static(parsed.path)
        except Exception as exc:
            self.send_error_json(exc)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            body = json.loads(raw or "{}")
            if parsed.path == "/api/query":
                catalog = get_catalog()
                query = str(body.get("query") or "分析 PM2.5 的时空格局")
                variable = clean_variable(body.get("variable") or query, catalog)
                region = str(body.get("region") or "china")
                target_date = str(body.get("date") or "") or None
                analysis = build_analysis(variable, region=region, query=query, target_date=target_date)
                self.send_json({"llm": analysis["llm"], "selection": analysis["selection"]})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_error_json(exc)

    def serve_static(self, request_path: str) -> None:
        clean_path = unquote(request_path)
        if clean_path == "/china.json":
            target = CHINA_GEOJSON.resolve()
        elif clean_path.startswith("/03_llm_core_frontend/"):
            suffix = clean_path.removeprefix("/03_llm_core_frontend/")
            target = (FRONTEND_DIR / suffix).resolve()
        else:
            safe = Path(clean_path.lstrip("/"))
            target = (PROJECT_ROOT / safe).resolve()
        if not str(target).startswith(str(PROJECT_ROOT)) or not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime = "text/plain; charset=utf-8"
        if target.suffix == ".js":
            mime = "text/javascript; charset=utf-8"
        elif target.suffix == ".css":
            mime = "text/css; charset=utf-8"
        elif target.suffix == ".html":
            mime = "text/html; charset=utf-8"
        elif target.suffix == ".svg":
            mime = "image/svg+xml"
        elif target.suffix == ".json":
            mime = "application/json; charset=utf-8"
        self.send_file(target, mime)

    def send_file(self, path: Path, mime: str) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(json_ready(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, exc: Exception) -> None:
        self.send_json(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=8),
            },
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


def main() -> None:
    host = os.getenv("GRIDVIS_HOST", DEFAULT_HOST)
    port = int(os.getenv("GRIDVIS_PORT", str(DEFAULT_PORT)))
    get_catalog()
    requested_port = port
    server = None
    for candidate in range(requested_port, requested_port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate), GridVisHandler)
            port = candidate
            break
        except OSError as exc:
            if candidate == requested_port + 19:
                raise
            if exc.errno not in {48, 98}:
                raise
    if server is None:
        raise RuntimeError("Unable to start GridVis-LLM server.")
    if port != requested_port:
        print(f"Port {requested_port} is busy; using http://{host}:{port}/ instead.")
    print(f"GridVis-LLM server running at http://{host}:{port}/frontend/final/index.html")
    print(f"Default workspace UI: http://{host}:{port}/")
    print(f"Compatibility UI: http://{host}:{port}/03_llm_core_frontend/index.html")
    print(f"Data directory: {DATA_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down GridVis-LLM server")


if __name__ == "__main__":
    main()
