from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


def encode_grid_transformer(
    grid_tensor: Any,
    *,
    patch_shape: tuple[int, int] = (32, 32),
    model_dim: int = 32,
    num_heads: int = 4,
    max_tokens: int = 384,
    query_text: str | None = None,
    include_saliency_map: bool = False,
) -> dict[str, Any]:
    """Build a lightweight Transformer representation for generic grid data.

    This is a dependency-free Transformer baseline: it tokenizes the grid into
    spatial patches, adds deterministic 2D positional encodings, runs one
    multi-head self-attention block, and returns patch saliency. It is designed
    as a local substitute for ViT/CLIP-style wiring and can be replaced by a
    pretrained model behind the same output schema.
    """

    features, metadata = make_patch_feature_matrix(grid_tensor, patch_shape=patch_shape, max_tokens=max_tokens)
    if features.size == 0:
        return empty_transformer_result(model_dim=model_dim, patch_shape=patch_shape)

    tokens = project_features(features, model_dim=model_dim)
    positions = np.asarray(
        [[item["center_y_norm"], item["center_x_norm"]] for item in metadata],
        dtype=float,
    )
    tokens = layer_norm(tokens + positional_encoding_2d(positions, model_dim=model_dim))
    encoded, attention = transformer_self_attention(tokens, num_heads=num_heads)
    global_token = layer_norm(np.mean(encoded, axis=0, keepdims=True))[0]
    saliency, saliency_sources = fused_patch_saliency(attention, encoded, metadata)
    patch_tokens = attach_patch_tokens(metadata, encoded, saliency, saliency_sources)

    result: dict[str, Any] = {
        "enabled": True,
        "model": {
            "name": "grid_transformer_numpy",
            "type": "deterministic_patch_self_attention",
            "model_dim": model_dim,
            "num_heads": num_heads,
            "patch_shape": list(patch_shape),
            "token_count": len(patch_tokens),
            "max_tokens": max_tokens,
        },
        "global_token": [safe_float(value) or 0.0 for value in global_token],
        "patch_tokens": patch_tokens,
        "top_salient_patches": sorted(
            patch_tokens,
            key=lambda item: item["saliency"],
            reverse=True,
        )[:10],
        "attention_edges": top_attention_edges(attention, metadata, limit=20),
        "saliency_summary": {
            "min": safe_float(np.nanmin(saliency)),
            "max": safe_float(np.nanmax(saliency)),
            "mean": safe_float(np.nanmean(saliency)),
            "p95": safe_float(np.nanpercentile(saliency, 95)),
            "method": "0.50 visual_distinctiveness + 0.30 attention_centrality + 0.20 token_feature_energy",
        },
        "saliency_method": {
            "visual_distinctiveness": ["gradient_strength", "local_texture", "dynamic_range", "entropy", "colorfulness"],
            "attention": "mean incoming multi-head self-attention",
            "embedding": "encoded token norm",
            "postprocess": "patch fill plus gaussian smoothing when saliency_map is requested",
        },
    }
    if query_text:
        result["query_alignment"] = align_query_to_patch_tokens(query_text, patch_tokens, model_dim=model_dim)
    if include_saliency_map:
        result["saliency_map"] = build_saliency_map(grid_tensor, patch_tokens)
    return result


def make_patch_feature_matrix(
    grid_tensor: Any,
    *,
    patch_shape: tuple[int, int],
    max_tokens: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    data = np.asarray(grid_tensor.data, dtype=float)
    valid_region = grid_tensor.mask.get("valid_region")
    height = data.shape[-2]
    width = data.shape[-1]
    patch_h, patch_w = choose_patch_shape(height, width, patch_shape, max_tokens=max_tokens)

    features = []
    metadata = []
    lat = np.asarray(grid_tensor.space["coordinates"]["lat"], dtype=float)
    lon = np.asarray(grid_tensor.space["coordinates"]["lon"], dtype=float)

    for y0 in range(0, height, patch_h):
        y1 = min(y0 + patch_h, height)
        for x0 in range(0, width, patch_w):
            x1 = min(x0 + patch_w, width)
            patch = data[:, :, y0:y1, x0:x1]
            if valid_region is not None:
                patch = np.where(valid_region[np.newaxis, np.newaxis, y0:y1, x0:x1], patch, np.nan)
            vector = patch_feature_vector(patch)
            if not vector:
                continue
            summary = patch_feature_summary(patch)
            features.append(vector)
            metadata.append(
                {
                    "token_id": len(metadata),
                    "bounds": {
                        "lat": [safe_float(lat[y0]), safe_float(lat[y1 - 1])],
                        "lon": [safe_float(lon[x0]), safe_float(lon[x1 - 1])],
                    },
                    "index_range": {"y": [int(y0), int(y1)], "x": [int(x0), int(x1)]},
                    "center_y_norm": ((y0 + y1 - 1) / 2) / max(1, height - 1),
                    "center_x_norm": ((x0 + x1 - 1) / 2) / max(1, width - 1),
                    "cell_count": int((y1 - y0) * (x1 - x0)),
                    "feature_summary": summary,
                }
            )
    return np.asarray(features, dtype=float), metadata


def choose_patch_shape(height: int, width: int, requested: tuple[int, int], *, max_tokens: int) -> tuple[int, int]:
    patch_h, patch_w = requested
    patch_h = max(1, int(patch_h))
    patch_w = max(1, int(patch_w))
    token_count = math.ceil(height / patch_h) * math.ceil(width / patch_w)
    if token_count <= max_tokens:
        return patch_h, patch_w
    scale = math.sqrt(token_count / max_tokens)
    return max(1, int(math.ceil(patch_h * scale))), max(1, int(math.ceil(patch_w * scale)))


def patch_feature_vector(patch: np.ndarray) -> list[float]:
    vector: list[float] = []
    for variable_idx in range(patch.shape[1]):
        values = patch[:, variable_idx]
        finite = np.isfinite(values)
        if not finite.any():
            vector.extend([0.0] * 8)
            continue
        flat = values[finite]
        mean_map = nanmean_time(values)
        vector.extend(
            [
                safe_stat(flat, np.nanmean),
                safe_stat(flat, np.nanstd),
                safe_stat(flat, np.nanmin),
                safe_stat(flat, np.nanmax),
                safe_float(np.nanpercentile(flat, 5)) or 0.0,
                safe_float(np.nanpercentile(flat, 95)) or 0.0,
                safe_ratio(np.count_nonzero(~finite), values.size),
                gradient_strength(mean_map),
            ]
        )
    return vector


def patch_feature_summary(patch: np.ndarray) -> dict[str, Any]:
    intensity = patch_intensity_map(patch)
    finite = np.isfinite(intensity)
    if not finite.any():
        return {
            "mean": None,
            "std": None,
            "dynamic_range": None,
            "gradient_strength": 0.0,
            "entropy": 0.0,
            "colorfulness": 0.0,
        }
    flat = intensity[finite]
    return {
        "mean": safe_float(np.nanmean(flat)),
        "std": safe_float(np.nanstd(flat)),
        "dynamic_range": safe_float(np.nanmax(flat) - np.nanmin(flat)),
        "gradient_strength": safe_float(gradient_strength(intensity)) or 0.0,
        "entropy": safe_float(histogram_entropy(flat)) or 0.0,
        "colorfulness": safe_float(patch_colorfulness(patch)) or 0.0,
    }


def patch_intensity_map(patch: np.ndarray) -> np.ndarray:
    # patch shape: time, variable, y, x. RGB image grids are stored as
    # intensity/red/green/blue, so the first variable is already luminance.
    if patch.shape[1] >= 1:
        return nanmean_time(patch[:, 0])
    return nanmean_time(np.nanmean(patch, axis=1))


def patch_colorfulness(patch: np.ndarray) -> float:
    if patch.shape[1] < 4:
        return 0.0
    red = nanmean_time(patch[:, 1])
    green = nanmean_time(patch[:, 2])
    blue = nanmean_time(patch[:, 3])
    finite = np.isfinite(red) & np.isfinite(green) & np.isfinite(blue)
    if not np.any(finite):
        return 0.0
    rg = red[finite] - green[finite]
    yb = 0.5 * (red[finite] + green[finite]) - blue[finite]
    return float(np.sqrt(np.nanstd(rg) ** 2 + np.nanstd(yb) ** 2) + 0.3 * np.sqrt(np.nanmean(rg) ** 2 + np.nanmean(yb) ** 2))


def project_features(features: np.ndarray, *, model_dim: int) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    col_mean = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    x = np.where(np.isfinite(x), x, col_mean)
    col_std = np.nanstd(x, axis=0)
    col_std = np.where(col_std > 1.0e-12, col_std, 1.0)
    x = (x - col_mean) / col_std

    weights = deterministic_matrix(x.shape[1], model_dim, "feature_projection")
    return x @ weights / math.sqrt(max(1, x.shape[1]))


def positional_encoding_2d(positions: np.ndarray, *, model_dim: int) -> np.ndarray:
    enc = np.zeros((positions.shape[0], model_dim), dtype=float)
    half = max(1, model_dim // 2)
    for axis in range(2):
        coord = positions[:, axis : axis + 1]
        start = axis * half
        end = min(model_dim, start + half)
        dim = end - start
        if dim <= 0:
            continue
        freqs = np.exp(np.arange(0, dim, 2, dtype=float) * (-math.log(10000.0) / max(1, dim)))
        values = coord * freqs[np.newaxis, :] * 2 * math.pi
        enc[:, start:end:2] = np.sin(values[:, : enc[:, start:end:2].shape[1]])
        if start + 1 < end:
            enc[:, start + 1 : end : 2] = np.cos(values[:, : enc[:, start + 1 : end : 2].shape[1]])
    return enc


def transformer_self_attention(tokens: np.ndarray, *, num_heads: int) -> tuple[np.ndarray, np.ndarray]:
    n_tokens, model_dim = tokens.shape
    num_heads = max(1, min(num_heads, model_dim))
    head_dim = max(1, model_dim // num_heads)
    usable_dim = head_dim * num_heads
    x = tokens[:, :usable_dim]

    wq = deterministic_matrix(usable_dim, usable_dim, "wq")
    wk = deterministic_matrix(usable_dim, usable_dim, "wk")
    wv = deterministic_matrix(usable_dim, usable_dim, "wv")
    wo = deterministic_matrix(usable_dim, usable_dim, "wo")
    q = (x @ wq).reshape(n_tokens, num_heads, head_dim).transpose(1, 0, 2)
    k = (x @ wk).reshape(n_tokens, num_heads, head_dim).transpose(1, 0, 2)
    v = (x @ wv).reshape(n_tokens, num_heads, head_dim).transpose(1, 0, 2)

    scores = q @ k.transpose(0, 2, 1) / math.sqrt(head_dim)
    scores = scores - np.nanmax(scores, axis=-1, keepdims=True)
    attention = np.exp(scores)
    attention = attention / np.maximum(np.sum(attention, axis=-1, keepdims=True), 1.0e-12)
    attended = (attention @ v).transpose(1, 0, 2).reshape(n_tokens, usable_dim)
    projected = attended @ wo / math.sqrt(usable_dim)

    output = np.zeros_like(tokens)
    output[:, :usable_dim] = projected
    if usable_dim < model_dim:
        output[:, usable_dim:] = tokens[:, usable_dim:]
    return layer_norm(tokens + output), attention


def attention_saliency(attention: np.ndarray, encoded: np.ndarray) -> np.ndarray:
    incoming = np.mean(attention, axis=(0, 1))
    norms = np.linalg.norm(encoded, axis=1)
    saliency = incoming * norms
    min_value = float(np.nanmin(saliency))
    max_value = float(np.nanmax(saliency))
    if max_value == min_value:
        return np.zeros_like(saliency)
    return (saliency - min_value) / (max_value - min_value)


def fused_patch_saliency(
    attention: np.ndarray,
    encoded: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    attention_score = normalize_vector(attention_saliency(attention, encoded))
    token_energy = token_feature_energy(metadata)
    visual = visual_distinctiveness(metadata)
    saliency = normalize_vector(0.50 * visual + 0.30 * attention_score + 0.20 * token_energy)
    sources = []
    for visual_score, attention_value, energy_value in zip(visual, attention_score, token_energy):
        sources.append(
            {
                "visual_distinctiveness": safe_float(visual_score) or 0.0,
                "attention_centrality": safe_float(attention_value) or 0.0,
                "token_feature_energy": safe_float(energy_value) or 0.0,
            }
        )
    return saliency, sources


def visual_distinctiveness(metadata: Sequence[Mapping[str, Any]]) -> np.ndarray:
    rows = []
    for item in metadata:
        summary = item.get("feature_summary") or {}
        rows.append(
            [
                float(summary.get("std") or 0.0),
                float(summary.get("dynamic_range") or 0.0),
                float(summary.get("gradient_strength") or 0.0),
                float(summary.get("entropy") or 0.0),
                float(summary.get("colorfulness") or 0.0),
            ]
        )
    if not rows:
        return np.empty(0, dtype=float)
    features = np.asarray(rows, dtype=float)
    features = np.column_stack([normalize_vector(features[:, idx]) for idx in range(features.shape[1])])
    weights = np.asarray([0.20, 0.20, 0.28, 0.18, 0.14], dtype=float)
    return normalize_vector(features @ weights)


def token_feature_energy(metadata: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = []
    for item in metadata:
        summary = item.get("feature_summary") or {}
        values.append(
            abs(float(summary.get("mean") or 0.0))
            + float(summary.get("std") or 0.0)
            + float(summary.get("dynamic_range") or 0.0)
            + float(summary.get("gradient_strength") or 0.0)
        )
    return normalize_vector(np.asarray(values, dtype=float))


def attach_patch_tokens(
    metadata: Sequence[Mapping[str, Any]],
    encoded: np.ndarray,
    saliency: np.ndarray,
    saliency_sources: Sequence[Mapping[str, float]] | None = None,
) -> list[dict[str, Any]]:
    tokens = []
    saliency_sources = saliency_sources or [{} for _ in range(len(metadata))]
    for item, vector, score, sources in zip(metadata, encoded, saliency, saliency_sources):
        enriched = dict(item)
        enriched["saliency"] = safe_float(score) or 0.0
        enriched["saliency_sources"] = dict(sources)
        enriched["vector"] = [safe_float(value) or 0.0 for value in vector]
        tokens.append(enriched)
    return tokens


def align_query_to_patch_tokens(query_text: str, patch_tokens: Sequence[Mapping[str, Any]], *, model_dim: int) -> dict[str, Any]:
    query_vector = text_hash_embedding(query_text, model_dim=model_dim)
    aligned = []
    for token in patch_tokens:
        vector = np.asarray(token.get("vector", []), dtype=float)
        if vector.size != model_dim:
            continue
        score = cosine_similarity(query_vector, vector)
        item = {
            "token_id": token.get("token_id"),
            "score": safe_float(score),
            "bounds": token.get("bounds"),
            "index_range": token.get("index_range"),
        }
        aligned.append(item)
    aligned.sort(key=lambda item: item["score"] if item["score"] is not None else -1.0, reverse=True)
    return {
        "query": query_text,
        "method": "hashed_text_to_grid_token_cosine",
        "top_aligned_patches": aligned[:10],
    }


def build_saliency_map(grid_tensor: Any, patch_tokens: Sequence[Mapping[str, Any]]) -> np.ndarray:
    height = int(grid_tensor.data.shape[-2])
    width = int(grid_tensor.data.shape[-1])
    saliency_map = np.zeros((height, width), dtype=float)
    for token in patch_tokens:
        ranges = token.get("index_range") or {}
        y0, y1 = ranges.get("y", [0, 0])
        x0, x1 = ranges.get("x", [0, 0])
        saliency_map[int(y0) : int(y1), int(x0) : int(x1)] = float(token.get("saliency") or 0.0)
    patch_heights = [int((token.get("index_range") or {}).get("y", [0, 0])[1]) - int((token.get("index_range") or {}).get("y", [0, 0])[0]) for token in patch_tokens]
    patch_widths = [int((token.get("index_range") or {}).get("x", [0, 0])[1]) - int((token.get("index_range") or {}).get("x", [0, 0])[0]) for token in patch_tokens]
    sigma = max(1.0, float(np.nanmedian(patch_heights + patch_widths)) / 7.0) if patch_tokens else 1.0
    from scipy import ndimage

    smoothed = ndimage.gaussian_filter(saliency_map, sigma=sigma)
    return normalize_map(smoothed)


def top_attention_edges(attention: np.ndarray, metadata: Sequence[Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if attention.size == 0:
        return []
    mean_attention = np.mean(attention, axis=0)
    np.fill_diagonal(mean_attention, 0.0)
    flat_indices = np.argsort(mean_attention.reshape(-1))[::-1][:limit]
    edges = []
    n = mean_attention.shape[0]
    for flat in flat_indices:
        src = int(flat // n)
        dst = int(flat % n)
        score = safe_float(mean_attention[src, dst])
        if score is None or score <= 0:
            continue
        edges.append(
            {
                "source_token": metadata[src].get("token_id"),
                "target_token": metadata[dst].get("token_id"),
                "attention": score,
            }
        )
    return edges


def text_hash_embedding(text: str, *, model_dim: int) -> np.ndarray:
    vector = np.zeros(model_dim, dtype=float)
    normalized = text.lower().strip()
    if not normalized:
        return vector
    grams = [normalized[idx : idx + 2] for idx in range(max(1, len(normalized) - 1))]
    for gram in grams:
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % model_dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def deterministic_matrix(rows: int, cols: int, namespace: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(namespace.encode("utf-8")).digest()[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(rows, cols))


def layer_norm(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mean = np.mean(arr, axis=-1, keepdims=True)
    std = np.std(arr, axis=-1, keepdims=True)
    return (arr - mean) / np.maximum(std, 1.0e-6)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def nanmean_time(values: np.ndarray) -> np.ndarray:
    finite_count = np.sum(np.isfinite(values), axis=0)
    return np.divide(
        np.nansum(values, axis=0),
        finite_count,
        out=np.full(values.shape[1:], np.nan, dtype=float),
        where=finite_count > 0,
    )


def gradient_strength(values: np.ndarray) -> float:
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2 or not np.isfinite(values).any():
        return 0.0
    filled = fill_nan_nearest(values)
    grad_y, grad_x = np.gradient(filled)
    gradient = np.hypot(grad_x, grad_y)
    finite = np.isfinite(values)
    return float(np.nanmean(gradient[finite])) if np.any(finite) else 0.0


def histogram_entropy(values: np.ndarray, *, bins: int = 32) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    hist, _ = np.histogram(finite, bins=bins)
    total = np.sum(hist)
    if total == 0:
        return 0.0
    prob = hist[hist > 0] / total
    return float(-np.sum(prob * np.log2(prob)) / math.log2(bins))


def normalize_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.where(np.isfinite(arr), arr, 0.0)
    min_value = float(np.nanmin(arr))
    max_value = float(np.nanmax(arr))
    if max_value == min_value:
        return np.zeros_like(arr)
    return (arr - min_value) / (max_value - min_value)


def normalize_map(values: np.ndarray) -> np.ndarray:
    return normalize_vector(np.asarray(values, dtype=float).reshape(-1)).reshape(values.shape)


def fill_nan_nearest(values: np.ndarray) -> np.ndarray:
    arr = values.astype(float, copy=True)
    missing = ~np.isfinite(arr)
    if not missing.any():
        return arr
    valid = ~missing
    if not valid.any():
        return np.zeros_like(arr)
    from scipy import ndimage

    _, indices = ndimage.distance_transform_edt(missing, return_indices=True)
    arr[missing] = arr[tuple(index[missing] for index in indices)]
    return arr


def safe_stat(values: np.ndarray, func: Any) -> float:
    if values.size == 0 or not np.isfinite(values).any():
        return 0.0
    return float(func(values))


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def safe_ratio(part: int, total: int) -> float:
    return float(part / total) if total else 0.0


def empty_transformer_result(*, model_dim: int, patch_shape: tuple[int, int]) -> dict[str, Any]:
    return {
        "enabled": False,
        "model": {
            "name": "grid_transformer_numpy",
            "type": "deterministic_patch_self_attention",
            "model_dim": model_dim,
            "patch_shape": list(patch_shape),
            "token_count": 0,
        },
        "global_token": [],
        "patch_tokens": [],
        "top_salient_patches": [],
        "attention_edges": [],
        "saliency_summary": {},
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return json_ready(value.item())
    return value
