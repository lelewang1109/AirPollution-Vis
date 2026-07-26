from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULES_DIR = PROJECT_ROOT / "modules"
DATA_ADAPTER_DIR = MODULES_DIR / "01_data_adapter"
if str(DATA_ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_ADAPTER_DIR))

from adapter import GridTensor, load_grid_data  # noqa: E402
from transformer_grid_encoder import encode_grid_transformer  # noqa: E402


def extract_grid_features(
    grid: GridTensor | Mapping[str, Any] | str | Path | Sequence[str | Path],
    *,
    variables: Sequence[str] | None = None,
    block_shape: tuple[int, int] = (32, 32),
    embedding_dim: int = 8,
    hotspot_quantile: float = 0.9,
    fill_missing: bool = True,
) -> dict[str, Any]:
    grid_tensor, source_summary = coerce_grid_tensor(grid, variables=variables, fill_missing=fill_missing)
    selected = filter_grid_tensor(grid_tensor, variables)

    global_statistics = extract_global_statistics(selected)
    spatial_features = extract_spatial_features(selected, hotspot_quantile=hotspot_quantile)
    temporal_features = extract_temporal_features(selected)
    embeddings = build_embeddings(selected, block_shape=block_shape, embedding_dim=embedding_dim)
    image_features = extract_image_features(selected)
    transformer_features = encode_grid_transformer(
        selected,
        patch_shape=block_shape,
        model_dim=max(16, embedding_dim * 4),
        num_heads=4,
        include_saliency_map=False,
    )
    semantic_summary = build_semantic_summary(
        selected,
        global_statistics=global_statistics,
        spatial_features=spatial_features,
        temporal_features=temporal_features,
        image_features=image_features,
        transformer_features=transformer_features,
    )

    return {
        "global_statistics": global_statistics,
        "spatial_features": spatial_features,
        "temporal_features": temporal_features,
        "image_features": image_features,
        "transformer_features": transformer_features,
        "embeddings": embeddings,
        "semantic_summary": semantic_summary,
        "metadata": {
            "source_summary": source_summary,
            "dimension_order": list(selected.dimension_order),
            "tensor_shape": list(selected.data.shape),
            "block_shape": list(block_shape),
            "embedding_dim": embedding_dim,
            "hotspot_quantile": hotspot_quantile,
        },
    }


def coerce_grid_tensor(
    grid: GridTensor | Mapping[str, Any] | str | Path | Sequence[str | Path],
    *,
    variables: Sequence[str] | None,
    fill_missing: bool,
) -> tuple[GridTensor, dict[str, Any]]:
    if isinstance(grid, GridTensor):
        return grid, grid.summary()
    if isinstance(grid, Mapping) and isinstance(grid.get("grid_tensor"), GridTensor):
        tensor = grid["grid_tensor"]
        return tensor, {
            "grid_spec": summarize_grid_spec(grid.get("grid_spec")),
            "spatiotemporal_summary": json_ready(grid.get("spatiotemporal_summary")),
        }

    loaded = load_grid_data(grid, variables=variables, fill_missing=fill_missing)
    return loaded["grid_tensor"], {
        "grid_spec": summarize_grid_spec(loaded.get("grid_spec")),
        "spatiotemporal_summary": json_ready(loaded.get("spatiotemporal_summary")),
    }


def filter_grid_tensor(grid_tensor: GridTensor, variables: Sequence[str] | None) -> GridTensor:
    if not variables:
        return grid_tensor

    missing = [name for name in variables if name not in grid_tensor.variables]
    if missing:
        raise ValueError(f"Variables not present in GridTensor: {missing}")

    indices = [grid_tensor.variables.index(name) for name in variables]
    return GridTensor(
        data=grid_tensor.data[:, indices].copy(),
        space=grid_tensor.space,
        time=grid_tensor.time,
        variables=[grid_tensor.variables[idx] for idx in indices],
        resolution=grid_tensor.resolution,
        mask={
            key: value[:, indices].copy() if value.ndim == 4 else value.copy()
            for key, value in grid_tensor.mask.items()
        },
        metadata=grid_tensor.metadata,
        variable_units={name: grid_tensor.variable_units.get(name) for name in variables},
        dimension_order=grid_tensor.dimension_order,
    )


def extract_global_statistics(grid_tensor: GridTensor) -> dict[str, Any]:
    data = grid_tensor.data
    original_missing = grid_tensor.mask.get("original_missing", ~np.isfinite(data))
    variable_statistics = {}

    for idx, name in enumerate(grid_tensor.variables):
        values = data[:, idx]
        finite = np.isfinite(values)
        variable_statistics[name] = {
            "min": safe_float(np.nanmin(values)) if finite.any() else None,
            "max": safe_float(np.nanmax(values)) if finite.any() else None,
            "mean": safe_float(np.nanmean(values)) if finite.any() else None,
            "std": safe_float(np.nanstd(values)) if finite.any() else None,
            "median": safe_float(np.nanmedian(values)) if finite.any() else None,
            "p05": safe_float(np.nanpercentile(values, 5)) if finite.any() else None,
            "p95": safe_float(np.nanpercentile(values, 95)) if finite.any() else None,
            "missing_ratio": safe_ratio(np.count_nonzero(original_missing[:, idx]), original_missing[:, idx].size),
            "unit": grid_tensor.variable_units.get(name),
        }

    correlation = compute_variable_correlation(data, grid_tensor.variables)
    overall = flatten_finite(data)
    primary_name = grid_tensor.variables[0] if grid_tensor.variables else None
    primary = variable_statistics.get(primary_name, {})

    return {
        "primary_variable": primary_name,
        "min": primary.get("min"),
        "max": primary.get("max"),
        "mean": primary.get("mean"),
        "std": primary.get("std"),
        "missing_ratio": primary.get("missing_ratio"),
        "overall": {
            "min": safe_float(np.nanmin(overall)) if overall.size else None,
            "max": safe_float(np.nanmax(overall)) if overall.size else None,
            "mean": safe_float(np.nanmean(overall)) if overall.size else None,
            "std": safe_float(np.nanstd(overall)) if overall.size else None,
        },
        "variables": variable_statistics,
        "correlation": correlation,
    }


def compute_variable_correlation(data: np.ndarray, variables: Sequence[str]) -> dict[str, dict[str, float | None]]:
    correlations: dict[str, dict[str, float | None]] = {}
    flattened = [data[:, idx].reshape(-1) for idx in range(len(variables))]

    for i, left in enumerate(variables):
        correlations[left] = {}
        for j, right in enumerate(variables):
            x = flattened[i]
            y = flattened[j]
            valid = np.isfinite(x) & np.isfinite(y)
            if np.count_nonzero(valid) < 2:
                value = None
            else:
                x_valid = x[valid]
                y_valid = y[valid]
                if np.nanstd(x_valid) == 0 or np.nanstd(y_valid) == 0:
                    value = None
                else:
                    value = safe_float(np.corrcoef(x_valid, y_valid)[0, 1])
            correlations[left][right] = value
    return correlations


def extract_spatial_features(grid_tensor: GridTensor, *, hotspot_quantile: float) -> dict[str, Any]:
    lon = np.asarray(grid_tensor.space["coordinates"]["lon"], dtype=float)
    lat = np.asarray(grid_tensor.space["coordinates"]["lat"], dtype=float)
    valid_region = grid_tensor.mask.get("valid_region")
    by_variable = {}

    for idx, name in enumerate(grid_tensor.variables):
        values = grid_tensor.data[:, idx]
        mean_map = nanmean_time(values)
        if valid_region is not None:
            mean_map = np.where(valid_region, mean_map, np.nan)
        by_variable[name] = spatial_features_for_map(
            mean_map,
            lon=lon,
            lat=lat,
            hotspot_quantile=hotspot_quantile,
        )

    primary_name = grid_tensor.variables[0] if grid_tensor.variables else None
    primary = by_variable.get(primary_name, {})
    aggregate = aggregate_spatial_features(by_variable)

    return {
        "primary_variable": primary_name,
        "hotspot_count": primary.get("hotspot_count"),
        "hotspot_locations": primary.get("hotspot_locations"),
        "gradient_strength": primary.get("gradient_strength"),
        "anisotropy": primary.get("anisotropy"),
        "dominant_gradient_direction": primary.get("dominant_gradient_direction"),
        "connected_domains": primary.get("connected_domains"),
        "boundary_change": primary.get("boundary_change"),
        "by_variable": by_variable,
        "aggregate": aggregate,
    }


def spatial_features_for_map(
    values: np.ndarray,
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    hotspot_quantile: float,
) -> dict[str, Any]:
    finite = np.isfinite(values)
    if not finite.any():
        return {
            "hotspot_count": 0,
            "hotspot_locations": [],
            "gradient_strength": None,
            "anisotropy": None,
            "dominant_gradient_direction": None,
            "connected_domains": 0,
            "boundary_change": None,
            "threshold": None,
        }

    threshold = float(np.nanquantile(values[finite], hotspot_quantile))
    hotspot_mask = finite & (values >= threshold)
    labeled, hotspot_count = ndimage.label(hotspot_mask)
    locations = hotspot_locations(labeled, values, lon=lon, lat=lat)

    filled = fill_nan_for_spatial(values)
    grad_y, grad_x = np.gradient(filled)
    gradient_mag = np.hypot(grad_x, grad_y)
    gradient_valid = gradient_mag[finite]

    gx_valid = grad_x[finite]
    gy_valid = grad_y[finite]
    x_energy = float(np.nanmean(gx_valid**2)) if gx_valid.size else 0.0
    y_energy = float(np.nanmean(gy_valid**2)) if gy_valid.size else 0.0
    anisotropy = abs(x_energy - y_energy) / (x_energy + y_energy) if (x_energy + y_energy) else 0.0
    direction = gradient_direction_label(x_energy=x_energy, y_energy=y_energy)

    boundary = ndimage.binary_dilation(hotspot_mask) ^ ndimage.binary_erosion(hotspot_mask)
    boundary = boundary & finite
    boundary_change = safe_float(np.nanmean(gradient_mag[boundary])) if np.any(boundary) else None

    high_domain_mask = finite & (values >= np.nanmean(values[finite]) + np.nanstd(values[finite]))
    _, connected_domains = ndimage.label(high_domain_mask)

    return {
        "hotspot_count": int(hotspot_count),
        "hotspot_locations": locations,
        "gradient_strength": safe_float(np.nanmean(gradient_valid)) if gradient_valid.size else None,
        "gradient_strength_p95": safe_float(np.nanpercentile(gradient_valid, 95)) if gradient_valid.size else None,
        "anisotropy": safe_float(anisotropy),
        "dominant_gradient_direction": direction,
        "connected_domains": int(connected_domains),
        "boundary_change": boundary_change,
        "threshold": safe_float(threshold),
    }


def hotspot_locations(labeled: np.ndarray, values: np.ndarray, *, lon: np.ndarray, lat: np.ndarray) -> list[dict[str, Any]]:
    locations = []
    for label_id in range(1, int(labeled.max()) + 1):
        mask = labeled == label_id
        if not np.any(mask):
            continue
        yy, xx = np.where(mask)
        peak_idx = np.nanargmax(np.where(mask, values, np.nan))
        peak_y, peak_x = np.unravel_index(peak_idx, values.shape)
        locations.append(
            {
                "component": label_id,
                "cell_count": int(mask.sum()),
                "centroid_lon": safe_float(np.nanmean(lon[xx])),
                "centroid_lat": safe_float(np.nanmean(lat[yy])),
                "peak_lon": safe_float(lon[peak_x]),
                "peak_lat": safe_float(lat[peak_y]),
                "peak_value": safe_float(values[peak_y, peak_x]),
            }
        )
    locations.sort(key=lambda item: item["cell_count"], reverse=True)
    return locations[:10]


def aggregate_spatial_features(by_variable: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    keys = ["hotspot_count", "gradient_strength", "anisotropy", "connected_domains", "boundary_change"]
    aggregate = {}
    for key in keys:
        values = [item.get(key) for item in by_variable.values()]
        numeric = [float(value) for value in values if value is not None and np.isfinite(float(value))]
        aggregate[f"mean_{key}"] = safe_float(np.mean(numeric)) if numeric else None
    return aggregate


def extract_temporal_features(grid_tensor: GridTensor) -> dict[str, Any]:
    by_variable = {}
    for idx, name in enumerate(grid_tensor.variables):
        series = spatial_mean_series(grid_tensor.data[:, idx], grid_tensor.mask.get("valid_region"))
        by_variable[name] = temporal_features_for_series(series, grid_tensor.time)

    primary_name = grid_tensor.variables[0] if grid_tensor.variables else None
    primary = by_variable.get(primary_name, {})

    return {
        "primary_variable": primary_name,
        "trend": primary.get("trend"),
        "change_points": primary.get("change_points"),
        "periodicity": primary.get("periodicity"),
        "volatility": primary.get("volatility"),
        "by_variable": by_variable,
        "aggregate": aggregate_temporal_features(by_variable),
    }


def temporal_features_for_series(series: np.ndarray, time: Sequence[str | None]) -> dict[str, Any]:
    finite = np.isfinite(series)
    valid_series = series[finite]
    valid_time = [time[idx] for idx, ok in enumerate(finite) if ok]

    if valid_series.size == 0:
        return {
            "trend": "no_data",
            "slope": None,
            "change_points": [],
            "periodicity": "unknown",
            "volatility": None,
            "series_mean": None,
            "series_range": None,
        }
    if valid_series.size == 1:
        return {
            "trend": "single_time_step",
            "slope": 0.0,
            "change_points": [],
            "periodicity": "single_time_step",
            "volatility": 0.0,
            "series_mean": safe_float(valid_series[0]),
            "series_range": 0.0,
        }

    x = np.arange(valid_series.size, dtype=float)
    slope = float(np.polyfit(x, valid_series, deg=1)[0])
    trend = classify_trend(valid_series, slope)
    change_points = detect_change_points(valid_series, valid_time)
    periodicity = detect_periodicity(valid_series, valid_time)
    volatility = safe_float(np.nanstd(np.diff(valid_series)))

    return {
        "trend": trend,
        "slope": safe_float(slope),
        "change_points": change_points,
        "periodicity": periodicity,
        "volatility": volatility,
        "series_mean": safe_float(np.nanmean(valid_series)),
        "series_range": safe_float(np.nanmax(valid_series) - np.nanmin(valid_series)),
        "first_value": safe_float(valid_series[0]),
        "last_value": safe_float(valid_series[-1]),
    }


def classify_trend(series: np.ndarray, slope: float) -> str:
    value_range = float(np.nanmax(series) - np.nanmin(series))
    threshold = max(value_range * 0.02, abs(float(np.nanmean(series))) * 0.001, 1.0e-12)
    total_delta = slope * max(1, series.size - 1)
    if total_delta > threshold:
        return "increase"
    if total_delta < -threshold:
        return "decrease"
    return "stable"


def detect_change_points(series: np.ndarray, time: Sequence[str | None]) -> list[str | None]:
    if series.size < 3:
        return []
    diffs = np.diff(series)
    scale = float(np.nanstd(diffs))
    if scale == 0 or not np.isfinite(scale):
        return []
    threshold = max(2.0 * scale, np.nanpercentile(np.abs(diffs), 90))
    indices = np.where(np.abs(diffs) >= threshold)[0] + 1
    return [time[idx] if idx < len(time) else str(idx) for idx in indices[:10]]


def detect_periodicity(series: np.ndarray, time: Sequence[str | None]) -> str:
    if series.size < 6 or np.nanstd(series) == 0:
        return "insufficient_data"

    x = np.arange(series.size, dtype=float)
    trend = np.polyval(np.polyfit(x, series, deg=1), x)
    centered = series - trend
    centered = centered - np.nanmean(centered)
    if np.nanstd(centered) == 0:
        return "none_detected"
    autocorr = np.correlate(centered, centered, mode="full")[series.size - 1 :]
    if autocorr[0] == 0:
        return "insufficient_data"
    autocorr = autocorr / autocorr[0]
    max_lag = min(series.size // 2, 366)
    if max_lag < 2:
        return "insufficient_data"
    search = autocorr[2 : max_lag + 1]
    peaks = [
        idx + 2
        for idx in range(len(search))
        if (idx == 0 or search[idx] >= search[idx - 1]) and (idx == len(search) - 1 or search[idx] >= search[idx + 1])
    ]
    lag = int(max(peaks, key=lambda item: autocorr[item])) if peaks else int(np.nanargmax(search) + 2)
    strength = float(autocorr[lag])
    if strength < 0.3:
        return "none_detected"

    timedeltas = parsed_time_deltas_days(time)
    lag_days = lag * timedeltas if timedeltas else None
    if lag_days is None:
        return f"lag_{lag}_steps"
    if 0.8 <= lag_days <= 1.2:
        return "daily"
    if 6 <= lag_days <= 8:
        return "weekly"
    if 27 <= lag_days <= 32:
        return "monthly"
    if 80 <= lag_days <= 100:
        return "seasonal"
    if 350 <= lag_days <= 380:
        return "annual"
    return f"lag_{lag}_steps"


def aggregate_temporal_features(by_variable: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    volatility = [
        item.get("volatility")
        for item in by_variable.values()
        if item.get("volatility") is not None and np.isfinite(float(item["volatility"]))
    ]
    trend_counts: dict[str, int] = {}
    for item in by_variable.values():
        trend = str(item.get("trend", "unknown"))
        trend_counts[trend] = trend_counts.get(trend, 0) + 1
    return {
        "mean_volatility": safe_float(np.mean(volatility)) if volatility else None,
        "trend_counts": trend_counts,
    }


def build_embeddings(
    grid_tensor: GridTensor,
    *,
    block_shape: tuple[int, int],
    embedding_dim: int,
) -> dict[str, Any]:
    block_features, block_metadata = make_block_feature_matrix(grid_tensor, block_shape=block_shape)
    time_features, time_metadata = make_time_feature_matrix(grid_tensor)
    relation_features, relation_metadata = make_relation_feature_matrix(grid_tensor)

    block_embedding = attach_vectors(block_metadata, pca_embed(block_features, embedding_dim))
    time_slice_embedding = attach_vectors(time_metadata, pca_embed(time_features, embedding_dim))
    relation_embedding = attach_vectors(relation_metadata, pca_embed(relation_features, embedding_dim))

    return {
        "block_embedding": block_embedding,
        "time_slice_embedding": time_slice_embedding,
        "variable_relation_embedding": relation_embedding,
        "embedding_method": {
            "name": "deterministic_svd_projection",
            "dimension": embedding_dim,
            "normalization": "z_score_per_feature",
        },
    }


def make_block_feature_matrix(
    grid_tensor: GridTensor,
    *,
    block_shape: tuple[int, int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    lat = np.asarray(grid_tensor.space["coordinates"]["lat"], dtype=float)
    lon = np.asarray(grid_tensor.space["coordinates"]["lon"], dtype=float)
    block_lat, block_lon = block_shape
    features = []
    metadata = []

    for y0 in range(0, len(lat), block_lat):
        y1 = min(y0 + block_lat, len(lat))
        for x0 in range(0, len(lon), block_lon):
            x1 = min(x0 + block_lon, len(lon))
            block = grid_tensor.data[:, :, y0:y1, x0:x1]
            block_missing = grid_tensor.mask["original_missing"][:, :, y0:y1, x0:x1]
            vector = []
            for vidx in range(len(grid_tensor.variables)):
                values = block[:, vidx]
                finite = np.isfinite(values)
                vector.extend(
                    [
                        safe_stat(values, np.nanmean),
                        safe_stat(values, np.nanstd),
                        safe_stat(values, np.nanmin),
                        safe_stat(values, np.nanmax),
                        safe_ratio(np.count_nonzero(block_missing[:, vidx]), block_missing[:, vidx].size),
                        safe_stat_gradient(values),
                    ]
                )
            vector.extend(
                [
                    normalize_coord(np.nanmean(lat[y0:y1]), lat),
                    normalize_coord(np.nanmean(lon[x0:x1]), lon),
                    (y1 - y0) / max(1, len(lat)),
                    (x1 - x0) / max(1, len(lon)),
                ]
            )
            features.append(vector)
            metadata.append(
                {
                    "block_id": len(metadata),
                    "lat_index_range": [int(y0), int(y1)],
                    "lon_index_range": [int(x0), int(x1)],
                    "bounds": {
                        "lat": [safe_float(lat[y0]), safe_float(lat[y1 - 1])],
                        "lon": [safe_float(lon[x0]), safe_float(lon[x1 - 1])],
                    },
                }
            )

    return np.asarray(features, dtype=float), metadata


def make_time_feature_matrix(grid_tensor: GridTensor) -> tuple[np.ndarray, list[dict[str, Any]]]:
    features = []
    metadata = []
    valid_region = grid_tensor.mask.get("valid_region")
    for tidx, timestamp in enumerate(grid_tensor.time):
        vector = []
        for vidx in range(len(grid_tensor.variables)):
            values = grid_tensor.data[tidx, vidx]
            if valid_region is not None:
                values = np.where(valid_region, values, np.nan)
            vector.extend(
                [
                    safe_stat(values, np.nanmean),
                    safe_stat(values, np.nanstd),
                    safe_stat(values, np.nanmin),
                    safe_stat(values, np.nanmax),
                    safe_stat_gradient(values[np.newaxis, ...]),
                ]
            )
        features.append(vector)
        metadata.append({"time_index": tidx, "time": timestamp})
    return np.asarray(features, dtype=float), metadata


def make_relation_feature_matrix(grid_tensor: GridTensor) -> tuple[np.ndarray, list[dict[str, Any]]]:
    features = []
    metadata = []
    data = grid_tensor.data
    variables = grid_tensor.variables

    if len(variables) == 1:
        values = flatten_finite(data[:, 0])
        features.append(
            [
                safe_stat(values, np.nanmean),
                safe_stat(values, np.nanstd),
                1.0,
                0.0,
                0.0,
            ]
        )
        metadata.append({"relation": variables[0], "left": variables[0], "right": variables[0]})
        return np.asarray(features, dtype=float), metadata

    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            left = data[:, i].reshape(-1)
            right = data[:, j].reshape(-1)
            valid = np.isfinite(left) & np.isfinite(right)
            if np.count_nonzero(valid) >= 2:
                corr = np.corrcoef(left[valid], right[valid])[0, 1]
                diff = left[valid] - right[valid]
                ratio = safe_divide(np.nanmean(left[valid]), np.nanmean(right[valid]))
                mean_abs_diff = np.nanmean(np.abs(diff))
                std_diff = np.nanstd(diff)
            else:
                corr = np.nan
                ratio = np.nan
                mean_abs_diff = np.nan
                std_diff = np.nan
            features.append([corr, ratio, mean_abs_diff, std_diff, np.count_nonzero(valid)])
            metadata.append(
                {
                    "relation": f"{variables[i]}__{variables[j]}",
                    "left": variables[i],
                    "right": variables[j],
                    "correlation": safe_float(corr),
                }
            )
    return np.asarray(features, dtype=float), metadata


def pca_embed(features: np.ndarray, embedding_dim: int) -> np.ndarray:
    if features.size == 0:
        return np.empty((0, embedding_dim), dtype=float)

    x = np.asarray(features, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    col_mean = np.nanmean(x, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    x = np.where(np.isfinite(x), x, col_mean)
    col_std = np.nanstd(x, axis=0)
    col_std = np.where(col_std > 0, col_std, 1.0)
    x = (x - col_mean) / col_std

    n_rows, n_cols = x.shape
    out_dim = min(embedding_dim, n_rows, n_cols)
    embedding = np.zeros((n_rows, embedding_dim), dtype=float)
    if out_dim == 0:
        return embedding
    if n_rows == 1:
        embedding[0, : min(embedding_dim, n_cols)] = x[0, : min(embedding_dim, n_cols)]
        return embedding

    _, _, vt = np.linalg.svd(x, full_matrices=False)
    embedding[:, :out_dim] = x @ vt[:out_dim].T
    return embedding


def attach_vectors(metadata: list[dict[str, Any]], vectors: np.ndarray) -> list[dict[str, Any]]:
    attached = []
    for item, vector in zip(metadata, vectors):
        enriched = dict(item)
        enriched["vector"] = [safe_float(value) or 0.0 for value in vector]
        attached.append(enriched)
    return attached


def extract_image_features(grid_tensor: GridTensor) -> dict[str, Any]:
    if not is_image_grid(grid_tensor):
        return {"is_image_grid": False}

    intensity = image_intensity_map(grid_tensor)
    finite = np.isfinite(intensity)
    if not finite.any():
        return {
            "is_image_grid": True,
            "edge_density": None,
            "texture_mean": None,
            "histogram": [],
            "channel_summary": {},
        }

    filled = fill_nan_for_spatial(intensity)
    grad_y, grad_x = np.gradient(filled)
    gradient = np.hypot(grad_x, grad_y)
    edge_threshold = float(np.nanpercentile(gradient[finite], 90))
    edge_mask = finite & (gradient >= edge_threshold)

    local_mean = ndimage.uniform_filter(filled, size=5)
    local_mean_sq = ndimage.uniform_filter(filled * filled, size=5)
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean * local_mean, 0.0))
    entropy_map = local_entropy_map(filled, bins=16, size=7)
    hist_counts, hist_edges = np.histogram(intensity[finite], bins=16, range=(0, 255))
    color_features = image_color_features(grid_tensor)

    channel_summary = {}
    for idx, name in enumerate(grid_tensor.variables):
        values = grid_tensor.data[:, idx]
        channel_summary[name] = {
            "mean": safe_float(np.nanmean(values)),
            "std": safe_float(np.nanstd(values)),
            "min": safe_float(np.nanmin(values)),
            "max": safe_float(np.nanmax(values)),
        }

    height = int(grid_tensor.space["dimensions"]["lat"])
    width = int(grid_tensor.space["dimensions"]["lon"])
    return {
        "is_image_grid": True,
        "image_size": {"width": width, "height": height, "aspect_ratio": safe_float(width / height) if height else None},
        "channel_count": len(grid_tensor.variables),
        "channels": list(grid_tensor.variables),
        "intensity": {
            "mean": safe_float(np.nanmean(intensity[finite])),
            "std": safe_float(np.nanstd(intensity[finite])),
            "min": safe_float(np.nanmin(intensity[finite])),
            "max": safe_float(np.nanmax(intensity[finite])),
            "dynamic_range": safe_float(np.nanmax(intensity[finite]) - np.nanmin(intensity[finite])),
        },
        "edges": {
            "edge_density": safe_ratio(np.count_nonzero(edge_mask), np.count_nonzero(finite)),
            "edge_threshold_p90": safe_float(edge_threshold),
            "mean_edge_strength": safe_float(np.nanmean(gradient[finite])),
            "p95_edge_strength": safe_float(np.nanpercentile(gradient[finite], 95)),
        },
        "texture": {
            "local_std_mean": safe_float(np.nanmean(local_std[finite])),
            "local_std_p95": safe_float(np.nanpercentile(local_std[finite], 95)),
            "local_entropy_mean": safe_float(np.nanmean(entropy_map[finite])),
            "local_entropy_p95": safe_float(np.nanpercentile(entropy_map[finite], 95)),
            "texture_label": semantic_texture_label(local_std[finite]),
        },
        "color_space": color_features,
        "histogram": {
            "bins": [safe_float(value) for value in hist_edges],
            "counts": [int(value) for value in hist_counts],
        },
        "channel_summary": channel_summary,
    }


def is_image_grid(grid_tensor: GridTensor) -> bool:
    metadata = grid_tensor.metadata or {}
    dataset_attrs = metadata.get("dataset_attrs") or {}
    if metadata.get("grid_type") == "image" or dataset_attrs.get("grid_type") == "image":
        return True
    image_names = {"intensity", "red", "green", "blue", "alpha", "gray", "grayscale"}
    return bool(set(name.lower() for name in grid_tensor.variables) & image_names)


def image_intensity_map(grid_tensor: GridTensor) -> np.ndarray:
    names = [name.lower() for name in grid_tensor.variables]
    if "intensity" in names:
        idx = names.index("intensity")
        return nanmean_time(grid_tensor.data[:, idx])
    rgb_indices = [names.index(name) for name in ("red", "green", "blue") if name in names]
    if len(rgb_indices) == 3:
        red = nanmean_time(grid_tensor.data[:, rgb_indices[0]])
        green = nanmean_time(grid_tensor.data[:, rgb_indices[1]])
        blue = nanmean_time(grid_tensor.data[:, rgb_indices[2]])
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return nanmean_time(np.nanmean(grid_tensor.data, axis=1))


def image_color_features(grid_tensor: GridTensor) -> dict[str, Any]:
    names = [name.lower() for name in grid_tensor.variables]
    if not all(name in names for name in ("red", "green", "blue")):
        return {"available": False}
    red = nanmean_time(grid_tensor.data[:, names.index("red")])
    green = nanmean_time(grid_tensor.data[:, names.index("green")])
    blue = nanmean_time(grid_tensor.data[:, names.index("blue")])
    rgb = np.dstack([red, green, blue]) / 255.0
    hsv = rgb_to_hsv(rgb)
    finite = np.isfinite(hsv).all(axis=2)
    hue_hist, hue_edges = np.histogram(hsv[:, :, 0][finite], bins=12, range=(0, 1))
    saturation = hsv[:, :, 1][finite]
    value = hsv[:, :, 2][finite]
    colorfulness = colorfulness_metric(red, green, blue)
    return {
        "available": True,
        "hsv": {
            "mean_hue": circular_mean(hsv[:, :, 0][finite]),
            "mean_saturation": safe_float(np.nanmean(saturation)) if saturation.size else None,
            "p95_saturation": safe_float(np.nanpercentile(saturation, 95)) if saturation.size else None,
            "mean_value": safe_float(np.nanmean(value)) if value.size else None,
            "hue_histogram": {
                "bins": [safe_float(item) for item in hue_edges],
                "counts": [int(item) for item in hue_hist],
            },
        },
        "colorfulness": safe_float(colorfulness),
        "dominant_color_label": dominant_color_label(hue_hist),
    }


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=float)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    maxc = np.nanmax(arr, axis=2)
    minc = np.nanmin(arr, axis=2)
    delta = maxc - minc
    hue = np.zeros_like(maxc)
    mask = delta > 1.0e-12
    red_max = mask & (maxc == red)
    green_max = mask & (maxc == green)
    blue_max = mask & (maxc == blue)
    hue[red_max] = ((green[red_max] - blue[red_max]) / delta[red_max]) % 6
    hue[green_max] = ((blue[green_max] - red[green_max]) / delta[green_max]) + 2
    hue[blue_max] = ((red[blue_max] - green[blue_max]) / delta[blue_max]) + 4
    hue = hue / 6.0
    saturation = np.divide(delta, maxc, out=np.zeros_like(delta), where=maxc > 1.0e-12)
    return np.dstack([hue, saturation, maxc])


def local_entropy_map(values: np.ndarray, *, bins: int, size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    vmin = float(np.nanmin(arr[finite]))
    vmax = float(np.nanmax(arr[finite]))
    if vmax == vmin:
        return np.zeros_like(arr)
    quantized = np.clip(((arr - vmin) / (vmax - vmin) * (bins - 1)).astype(int), 0, bins - 1)
    entropy = np.zeros_like(arr, dtype=float)
    for label in range(bins):
        indicator = (quantized == label).astype(float)
        local_prob = ndimage.uniform_filter(indicator, size=size)
        positive = local_prob > 0
        contribution = np.zeros_like(local_prob)
        contribution[positive] = local_prob[positive] * np.log2(local_prob[positive])
        entropy -= contribution
    return entropy / np.log2(bins)


def colorfulness_metric(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> float:
    finite = np.isfinite(red) & np.isfinite(green) & np.isfinite(blue)
    if not np.any(finite):
        return 0.0
    rg = red[finite] - green[finite]
    yb = 0.5 * (red[finite] + green[finite]) - blue[finite]
    return float(np.sqrt(np.nanstd(rg) ** 2 + np.nanstd(yb) ** 2) + 0.3 * np.sqrt(np.nanmean(rg) ** 2 + np.nanmean(yb) ** 2))


def circular_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    angles = values * 2 * np.pi
    return safe_float((np.arctan2(np.nanmean(np.sin(angles)), np.nanmean(np.cos(angles))) % (2 * np.pi)) / (2 * np.pi))


def dominant_color_label(hue_hist: np.ndarray) -> str:
    if hue_hist.size == 0 or np.sum(hue_hist) == 0:
        return "unknown"
    labels = ["red", "orange", "yellow", "yellow-green", "green", "cyan-green", "cyan", "blue-cyan", "blue", "violet", "magenta", "red-magenta"]
    return labels[int(np.nanargmax(hue_hist)) % len(labels)]


def semantic_texture_label(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "unknown texture"
    mean_texture = float(np.nanmean(finite))
    if mean_texture < 5:
        return "smooth image regions"
    if mean_texture < 20:
        return "moderate local texture"
    return "high local texture or sharp detail"


def build_semantic_summary(
    grid_tensor: GridTensor,
    *,
    global_statistics: Mapping[str, Any],
    spatial_features: Mapping[str, Any],
    temporal_features: Mapping[str, Any],
    image_features: Mapping[str, Any] | None = None,
    transformer_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    primary = global_statistics.get("primary_variable")
    primary_stats = global_statistics.get("variables", {}).get(primary, {}) if primary else {}
    spatial = spatial_features.get("by_variable", {}).get(primary, {}) if primary else {}
    temporal = temporal_features.get("by_variable", {}).get(primary, {}) if primary else {}

    missing_ratio = primary_stats.get("missing_ratio")
    hotspot_count = spatial.get("hotspot_count")
    gradient = spatial.get("gradient_strength")
    trend = temporal.get("trend")
    strongest_relation = strongest_correlation(global_statistics.get("correlation", {}), primary)
    image_features = image_features or {}
    transformer_features = transformer_features or {}
    image_fact = None
    if image_features.get("is_image_grid"):
        size = image_features.get("image_size", {})
        edges = image_features.get("edges", {})
        texture = image_features.get("texture", {})
        image_fact = (
            f"Image grid: {size.get('width')} by {size.get('height')} pixels, "
            f"edge density {safe_float(edges.get('edge_density'))}, "
            f"texture {texture.get('texture_label')}."
        )
    llm_facts = [
        f"Primary variable: {primary}" if primary else "No primary variable detected.",
        f"Spatial grid: {grid_tensor.space['dimensions']['lat']} lat by {grid_tensor.space['dimensions']['lon']} lon cells.",
        f"Time coverage: {len(grid_tensor.time)} step(s), resolution {grid_tensor.resolution.get('time')}.",
        f"Hotspot components for {primary}: {hotspot_count}." if primary else "No hotspot summary available.",
        f"Trend for {primary}: {trend}." if primary else "No temporal trend available.",
    ]
    if image_fact:
        llm_facts.append(image_fact)
    if transformer_features.get("enabled"):
        model = transformer_features.get("model", {})
        saliency = transformer_features.get("saliency_summary", {})
        llm_facts.append(
            "Grid Transformer: "
            f"{model.get('token_count')} patch tokens, "
            f"{model.get('num_heads')} attention heads, "
            f"saliency p95 {safe_float(saliency.get('p95'))}."
        )

    return {
        "primary_variable": primary,
        "data_coverage": semantic_missing_label(missing_ratio),
        "spatial_pattern": semantic_spatial_label(hotspot_count, gradient, spatial.get("anisotropy")),
        "temporal_pattern": semantic_temporal_label(trend, temporal.get("volatility")),
        "image_pattern": image_features if image_features.get("is_image_grid") else None,
        "transformer_pattern": {
            "model": transformer_features.get("model"),
            "saliency_summary": transformer_features.get("saliency_summary"),
            "top_salient_patches": transformer_features.get("top_salient_patches", [])[:5],
        } if transformer_features.get("enabled") else None,
        "strongest_relation": strongest_relation,
        "llm_facts": llm_facts,
    }


def strongest_correlation(correlation: Mapping[str, Mapping[str, Any]], primary: str | None) -> dict[str, Any] | None:
    if not primary or primary not in correlation:
        return None
    best_name = None
    best_value = None
    for name, value in correlation[primary].items():
        if name == primary or value is None:
            continue
        if best_value is None or abs(float(value)) > abs(float(best_value)):
            best_name = name
            best_value = value
    if best_name is None:
        return None
    return {"variable": best_name, "correlation": safe_float(best_value)}


def semantic_missing_label(missing_ratio: float | None) -> str:
    if missing_ratio is None:
        return "unknown coverage"
    if missing_ratio == 0:
        return "complete grid"
    if missing_ratio < 0.05:
        return "nearly complete grid"
    if missing_ratio < 0.3:
        return "moderate missing regions"
    return "large fixed or invalid regions"


def semantic_spatial_label(hotspot_count: int | None, gradient: float | None, anisotropy: float | None) -> str:
    if hotspot_count is None:
        return "unknown spatial pattern"
    if hotspot_count == 0:
        return "no clear hotspots"
    spread = "directional" if anisotropy is not None and anisotropy > 0.35 else "diffuse"
    intensity = "sharp-gradient" if gradient is not None and gradient > 1 else "smooth-gradient"
    return f"{hotspot_count} {spread} hotspot region(s), {intensity}"


def semantic_temporal_label(trend: str | None, volatility: float | None) -> str:
    if trend in {None, "no_data"}:
        return "unknown temporal pattern"
    if trend == "single_time_step":
        return "single time slice"
    volatility_label = "volatile" if volatility is not None and volatility > 1 else "low volatility"
    return f"{trend}, {volatility_label}"


def nanmean_time(values: np.ndarray) -> np.ndarray:
    finite_count = np.sum(np.isfinite(values), axis=0)
    return np.divide(
        np.nansum(values, axis=0),
        finite_count,
        out=np.full(values.shape[1:], np.nan, dtype=float),
        where=finite_count > 0,
    )


def fill_nan_for_spatial(values: np.ndarray) -> np.ndarray:
    arr = values.astype(float, copy=True)
    missing = ~np.isfinite(arr)
    if not missing.any():
        return arr
    valid = ~missing
    if not valid.any():
        return np.zeros_like(arr)
    _, indices = ndimage.distance_transform_edt(missing, return_indices=True)
    arr[missing] = arr[tuple(index[missing] for index in indices)]
    return arr


def spatial_mean_series(values: np.ndarray, valid_region: np.ndarray | None) -> np.ndarray:
    arr = values.astype(float, copy=True)
    if valid_region is not None:
        arr = np.where(valid_region[np.newaxis, :, :], arr, np.nan)
    series = []
    for tidx in range(arr.shape[0]):
        finite = np.isfinite(arr[tidx])
        series.append(float(np.nanmean(arr[tidx])) if finite.any() else np.nan)
    return np.asarray(series, dtype=float)


def safe_stat(values: np.ndarray, func: Any) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return 0.0
    return float(func(arr))


def safe_stat_gradient(values: np.ndarray) -> float:
    if values.ndim == 3:
        mean_map = nanmean_time(values)
    else:
        mean_map = values
    if not np.isfinite(mean_map).any():
        return 0.0
    if mean_map.ndim != 2 or mean_map.shape[0] < 2 or mean_map.shape[1] < 2:
        return 0.0
    filled = fill_nan_for_spatial(mean_map)
    grad_y, grad_x = np.gradient(filled)
    return float(np.nanmean(np.hypot(grad_x, grad_y)))


def flatten_finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def gradient_direction_label(*, x_energy: float, y_energy: float) -> str:
    total = x_energy + y_energy
    if total == 0:
        return "flat"
    if x_energy > y_energy * 1.25:
        return "east-west"
    if y_energy > x_energy * 1.25:
        return "north-south"
    return "mixed"


def parsed_time_deltas_days(time: Sequence[str | None]) -> float | None:
    parsed = pd.to_datetime([item for item in time if item is not None], errors="coerce")
    parsed = parsed[~pd.isna(parsed)]
    if len(parsed) < 2:
        return None
    deltas = pd.Series(parsed).diff().dropna().dt.total_seconds().to_numpy()
    if deltas.size == 0:
        return None
    return float(np.nanmedian(deltas) / 86400.0)


def normalize_coord(value: float, coord: np.ndarray) -> float:
    coord = np.asarray(coord, dtype=float)
    coord_min = np.nanmin(coord)
    coord_max = np.nanmax(coord)
    if coord_max == coord_min:
        return 0.0
    return float((value - coord_min) / (coord_max - coord_min))


def safe_divide(left: float, right: float) -> float:
    if right == 0 or not np.isfinite(right):
        return np.nan
    return float(left / right)


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def safe_ratio(part: int, total: int) -> float:
    return float(part / total) if total else 0.0


def summarize_grid_spec(grid_spec: Any) -> Any:
    if not isinstance(grid_spec, Mapping):
        return json_ready(grid_spec)
    output = dict(grid_spec)
    coords = output.get("coordinates")
    if isinstance(coords, Mapping):
        output["coordinates"] = {}
        for name, values in coords.items():
            arr = np.asarray(values, dtype=float)
            output["coordinates"][name] = {
                "size": int(arr.size),
                "min": safe_float(np.nanmin(arr)) if arr.size else None,
                "max": safe_float(np.nanmax(arr)) if arr.size else None,
            }
    return json_ready(output)


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def cli_summary(result: Mapping[str, Any], *, max_blocks: int = 5) -> dict[str, Any]:
    output = json_ready(result)
    embeddings = output.get("embeddings", {})
    if "block_embedding" in embeddings:
        embeddings["block_embedding_preview"] = embeddings["block_embedding"][:max_blocks]
        embeddings["block_embedding_count"] = len(embeddings["block_embedding"])
        embeddings.pop("block_embedding", None)
    if "time_slice_embedding" in embeddings:
        embeddings["time_slice_embedding_preview"] = embeddings["time_slice_embedding"][:max_blocks]
        embeddings["time_slice_embedding_count"] = len(embeddings["time_slice_embedding"])
        embeddings.pop("time_slice_embedding", None)
    if "variable_relation_embedding" in embeddings:
        embeddings["variable_relation_embedding_preview"] = embeddings["variable_relation_embedding"][:max_blocks]
        embeddings["variable_relation_embedding_count"] = len(embeddings["variable_relation_embedding"])
        embeddings.pop("variable_relation_embedding", None)
    transformer = output.get("transformer_features", {})
    if "patch_tokens" in transformer:
        transformer["patch_tokens_preview"] = transformer["patch_tokens"][:max_blocks]
        transformer["patch_token_count"] = len(transformer["patch_tokens"])
        transformer.pop("patch_tokens", None)
    if "top_salient_patches" in transformer:
        transformer["top_salient_patches"] = transformer["top_salient_patches"][:max_blocks]
    return output


def parse_block_shape(value: str) -> tuple[int, int]:
    if "x" in value:
        left, right = value.lower().split("x", 1)
    elif "," in value:
        left, right = value.split(",", 1)
    else:
        left = right = value
    shape = (int(left), int(right))
    if shape[0] <= 0 or shape[1] <= 0:
        raise argparse.ArgumentTypeError("block shape must be positive")
    return shape


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract multiscale grid features from GridTensor-compatible data.")
    parser.add_argument("source", help="Input file, directory, or supported grid data path.")
    parser.add_argument("--variables", nargs="*", default=None, help="Optional variables to keep.")
    parser.add_argument("--block-shape", type=parse_block_shape, default=(32, 32), help="Block size, e.g. 32x32.")
    parser.add_argument("--embedding-dim", type=int, default=8, help="Embedding vector dimension.")
    parser.add_argument("--hotspot-quantile", type=float, default=0.9, help="Hotspot quantile threshold.")
    parser.add_argument("--no-fill", action="store_true", help="Keep missing values unfilled in Step 1.")
    args = parser.parse_args()

    result = extract_grid_features(
        args.source,
        variables=args.variables,
        block_shape=args.block_shape,
        embedding_dim=args.embedding_dim,
        hotspot_quantile=args.hotspot_quantile,
        fill_missing=not args.no_fill,
    )
    print(json.dumps(cli_summary(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
