from __future__ import annotations

import argparse
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from scipy import interpolate, ndimage


SUPPORTED_SUFFIXES = {".nc", ".netcdf", ".csv", ".tif", ".tiff", ".h5", ".hdf5", ".npy", ".npz", ".png"}
LON_ALIASES = ("lon", "longitude", "x", "xlon", "long", "lng")
LAT_ALIASES = ("lat", "latitude", "y", "ylat")
TIME_ALIASES = ("time", "date", "datetime", "timestamp")
FILL_VALUE_ATTRS = ("_FillValue", "missing_value", "fill_value", "nodata")


@dataclass
class GridTensor:
    data: np.ndarray
    space: dict[str, Any]
    time: list[str | None]
    variables: list[str]
    resolution: dict[str, Any]
    mask: dict[str, np.ndarray]
    metadata: dict[str, Any]
    variable_units: dict[str, str | None]
    dimension_order: tuple[str, str, str, str] = ("time", "variable", "lat", "lon")

    def to_xarray(self) -> xr.Dataset:
        coords = {
            "time": self.time,
            "variable": self.variables,
            "lat": self.space["coordinates"]["lat"],
            "lon": self.space["coordinates"]["lon"],
        }
        ds = xr.Dataset(
            {"values": (self.dimension_order, self.data)},
            coords=coords,
            attrs=self.metadata,
        )
        ds["original_missing_mask"] = (self.dimension_order, self.mask["original_missing"])
        ds["filled_mask"] = (self.dimension_order, self.mask["filled"])
        ds["fixed_missing_mask"] = (("lat", "lon"), self.mask["fixed_missing"])
        return ds

    def summary(self) -> dict[str, Any]:
        return {
            "shape": list(self.data.shape),
            "dimension_order": list(self.dimension_order),
            "variables": self.variables,
            "time": self.time,
            "resolution": self.resolution,
            "mask": {
                "original_missing_count": int(np.count_nonzero(self.mask["original_missing"])),
                "filled_count": int(np.count_nonzero(self.mask["filled"])),
                "fixed_missing_cell_count": int(np.count_nonzero(self.mask["fixed_missing"])),
            },
            "metadata": self.metadata,
        }


def load_grid_data(
    source: str | Path | np.ndarray | xr.Dataset | xr.DataArray | Sequence[str | Path],
    variables: Sequence[str] | None = None,
    *,
    coords: Mapping[str, Any] | None = None,
    fill_missing: bool = True,
    fill_method: str = "nearest",
    fixed_missing_threshold: float = 0.8,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ds, source_metadata = open_grid_dataset(source, coords=coords, metadata=metadata)
    standard = standardize_dataset(ds)
    grid_tensor = build_grid_tensor(
        standard,
        variables=variables,
        fill_missing=fill_missing,
        fill_method=fill_method,
        fixed_missing_threshold=fixed_missing_threshold,
        source_metadata=source_metadata,
    )

    variable_profiles = profile_variables(grid_tensor)
    grid_spec = make_grid_spec(grid_tensor)
    spatiotemporal_summary = summarize_spatiotemporal(grid_tensor, variable_profiles)

    return {
        "grid_spec": grid_spec,
        "grid_tensor": grid_tensor,
        "variable_profiles": variable_profiles,
        "spatiotemporal_summary": spatiotemporal_summary,
    }


def open_grid_dataset(
    source: str | Path | np.ndarray | xr.Dataset | xr.DataArray | Sequence[str | Path],
    *,
    coords: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[xr.Dataset, dict[str, Any]]:
    if isinstance(source, xr.Dataset):
        ds = source.copy()
        source_meta = {"source_type": "xarray.Dataset"}
    elif isinstance(source, xr.DataArray):
        name = source.name or "value"
        ds = source.to_dataset(name=name)
        source_meta = {"source_type": "xarray.DataArray"}
    elif isinstance(source, np.ndarray):
        ds = numpy_to_dataset(source, coords=coords)
        source_meta = {"source_type": "numpy.ndarray"}
    elif isinstance(source, (list, tuple)):
        paths = [Path(p) for p in source]
        ds = open_many_files(paths)
        source_meta = {"source_type": "file_sequence", "paths": [str(p) for p in paths]}
    else:
        path = Path(source)
        if path.is_dir():
            paths = sorted(p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
            ds = open_many_files(paths)
            source_meta = {"source_type": "directory", "path": str(path), "file_count": len(paths)}
        else:
            ds = open_one_file(path)
            source_meta = {"source_type": "file", "path": str(path), "suffix": path.suffix.lower()}

    if coords:
        ds = apply_coord_overrides(ds, coords)
    if metadata:
        source_meta.update(dict(metadata))
    return ds, source_meta


def open_one_file(path: Path) -> xr.Dataset:
    suffix = path.suffix.lower()
    if suffix in {".nc", ".netcdf"}:
        return xr.open_dataset(path, decode_times=True)
    if suffix == ".csv":
        return csv_to_dataset(path)
    if suffix in {".tif", ".tiff"}:
        return geotiff_to_dataset(path)
    if suffix == ".png":
        return image_to_dataset(path)
    if suffix in {".h5", ".hdf5"}:
        return hdf5_to_dataset(path)
    if suffix == ".npy":
        return numpy_to_dataset(np.load(path))
    if suffix == ".npz":
        loaded = np.load(path)
        arrays = {name: loaded[name] for name in loaded.files}
        return arrays_to_dataset(arrays)
    raise ValueError(f"Unsupported file suffix: {suffix}")


def open_many_files(paths: Sequence[Path]) -> xr.Dataset:
    if not paths:
        raise ValueError("No supported data files found.")

    datasets = []
    for path in paths:
        ds = open_one_file(path)
        standard = standardize_dataset(ds)
        if "time" not in standard.dims:
            scalar_time = standard.coords.get("time")
            if scalar_time is not None and scalar_time.ndim == 0:
                standard = standard.expand_dims(time=[pd.Timestamp(scalar_time.values).isoformat()])
            else:
                standard = standard.expand_dims(time=[infer_time_from_filename(path)])
        datasets.append(standard.load())
        ds.close()

    return xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal", compat="override")


def csv_to_dataset(path: Path) -> xr.Dataset:
    df = pd.read_csv(path)
    lon_col = find_column(df.columns, LON_ALIASES)
    lat_col = find_column(df.columns, LAT_ALIASES)
    time_col = find_column(df.columns, TIME_ALIASES)

    if not lon_col or not lat_col:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError("CSV must contain lon/lat columns or numeric matrix values.")
        arr = numeric.to_numpy(dtype=float)
        return numpy_to_dataset(arr, coords={"lat": np.arange(arr.shape[0]), "lon": np.arange(arr.shape[1])})

    value_cols = [
        col
        for col in df.columns
        if col not in {lon_col, lat_col, time_col} and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not value_cols:
        raise ValueError("CSV contains coordinates but no numeric grid variables.")

    lon = np.sort(df[lon_col].dropna().unique())
    lat = np.sort(df[lat_col].dropna().unique())
    coords: dict[str, Any] = {"lat": lat, "lon": lon}

    if time_col:
        time = pd.to_datetime(df[time_col], errors="ignore")
        unique_time = pd.Index(time).drop_duplicates().sort_values()
        coords["time"] = unique_time.astype(str).tolist()
        data_vars = {}
        for var in value_cols:
            arr = np.full((len(unique_time), len(lat), len(lon)), np.nan, dtype=float)
            for tidx, timestamp in enumerate(unique_time):
                sub = df.loc[time == timestamp, [lat_col, lon_col, var]]
                pivot = sub.pivot_table(index=lat_col, columns=lon_col, values=var, aggfunc="mean")
                arr[tidx] = pivot.reindex(index=lat, columns=lon).to_numpy(dtype=float)
            data_vars[var] = (("time", "lat", "lon"), arr)
    else:
        data_vars = {}
        for var in value_cols:
            pivot = df.pivot_table(index=lat_col, columns=lon_col, values=var, aggfunc="mean")
            data_vars[var] = (("lat", "lon"), pivot.reindex(index=lat, columns=lon).to_numpy(dtype=float))

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs={"source_format": "csv"})


def geotiff_to_dataset(path: Path) -> xr.Dataset:
    with rasterio.open(path) as src:
        data = src.read(masked=True).astype(float).filled(np.nan)
        band_names = list(src.descriptions or [])
        if not band_names or all(name is None for name in band_names):
            band_names = [f"band_{i}" for i in range(1, src.count + 1)]
        else:
            band_names = [name or f"band_{i}" for i, name in enumerate(band_names, start=1)]

        cols = np.arange(src.width)
        rows = np.arange(src.height)
        lon = np.array([rasterio.transform.xy(src.transform, 0, col, offset="center")[0] for col in cols])
        lat = np.array([rasterio.transform.xy(src.transform, row, 0, offset="center")[1] for row in rows])

        data_vars = {name: (("lat", "lon"), data[idx]) for idx, name in enumerate(band_names)}
        attrs = {
            "source_format": "geotiff",
            "crs": src.crs.to_string() if src.crs else None,
            "transform": tuple(src.transform),
            "nodata": src.nodata,
        }
        return xr.Dataset(data_vars=data_vars, coords={"lat": lat, "lon": lon}, attrs=attrs)


def image_to_dataset(path: Path) -> xr.Dataset:
    image = read_png(path)
    if image.ndim == 2:
        height, width = image.shape
        data_vars = {
            "intensity": (("lat", "lon"), image.astype(float), {"units": "digital_number"}),
        }
        channel_names = ["intensity"]
    else:
        height, width, channels = image.shape
        channel_names = ["red", "green", "blue", "alpha"][:channels]
        image_float = image.astype(float)
        data_vars = {}
        if channels >= 3:
            intensity = 0.2126 * image_float[:, :, 0] + 0.7152 * image_float[:, :, 1] + 0.0722 * image_float[:, :, 2]
            data_vars["intensity"] = (("lat", "lon"), intensity, {"units": "digital_number"})
        for idx, name in enumerate(channel_names):
            data_vars[name] = (("lat", "lon"), image_float[:, :, idx], {"units": "digital_number"})

    attrs = {
        "source_format": "png",
        "grid_type": "image",
        "coordinate_system": "pixel",
        "width": int(width),
        "height": int(height),
        "channels": channel_names,
        "bit_depth": 8,
        "crs": "pixel_coordinates",
    }
    return xr.Dataset(
        data_vars=data_vars,
        coords={"lat": np.arange(height), "lon": np.arange(width)},
        attrs=attrs,
    )


def read_png(path: Path) -> np.ndarray:
    data = path.read_bytes()
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        raise ValueError(f"Not a PNG file: {path}")

    offset = len(signature)
    width = height = bit_depth = color_type = interlace = None
    palette: list[tuple[int, int, int]] | None = None
    idat_parts = []

    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("Malformed PNG chunk header.")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if compression != 0 or filter_method != 0:
                raise ValueError("Unsupported PNG compression or filter method.")
        elif chunk_type == b"PLTE":
            palette = [
                tuple(chunk_data[idx : idx + 3])
                for idx in range(0, len(chunk_data), 3)
                if len(chunk_data[idx : idx + 3]) == 3
            ]
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if None in {width, height, bit_depth, color_type, interlace}:
        raise ValueError("PNG is missing IHDR metadata.")
    if bit_depth != 8:
        raise ValueError("Only 8-bit PNG images are currently supported.")
    if interlace != 0:
        raise ValueError("Interlaced PNG images are not currently supported.")

    channels_by_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if color_type not in channels_by_type:
        raise ValueError(f"Unsupported PNG color type: {color_type}")
    channels = channels_by_type[color_type]
    bpp = channels
    row_bytes = int(width) * channels
    raw = zlib.decompress(b"".join(idat_parts))
    expected = (row_bytes + 1) * int(height)
    if len(raw) < expected:
        raise ValueError("PNG pixel data is shorter than expected.")

    rows = []
    previous = bytearray(row_bytes)
    cursor = 0
    for _ in range(int(height)):
        filter_type = raw[cursor]
        cursor += 1
        scanline = bytearray(raw[cursor : cursor + row_bytes])
        cursor += row_bytes
        row = unfilter_png_scanline(scanline, previous, filter_type, bpp)
        rows.append(row)
        previous = row

    array = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape((int(height), int(width), channels))
    if color_type == 0:
        return array[:, :, 0]
    if color_type == 3:
        if palette is None:
            raise ValueError("Palette PNG is missing PLTE chunk.")
        rgb = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        palette_array = np.asarray(palette, dtype=np.uint8)
        indices = array[:, :, 0]
        valid = indices < len(palette_array)
        rgb[valid] = palette_array[indices[valid]]
        return rgb
    if color_type == 4:
        return array
    return array


def unfilter_png_scanline(scanline: bytearray, previous: bytearray, filter_type: int, bpp: int) -> bytearray:
    row = bytearray(scanline)
    for idx in range(len(row)):
        left = row[idx - bpp] if idx >= bpp else 0
        up = previous[idx] if idx < len(previous) else 0
        upper_left = previous[idx - bpp] if idx >= bpp and idx - bpp < len(previous) else 0
        if filter_type == 0:
            predictor = 0
        elif filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (left + up) // 2
        elif filter_type == 4:
            predictor = paeth_predictor(left, up, upper_left)
        else:
            raise ValueError(f"Unsupported PNG filter type: {filter_type}")
        row[idx] = (row[idx] + predictor) & 0xFF
    return row


def paeth_predictor(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    pa = abs(estimate - left)
    pb = abs(estimate - up)
    pc = abs(estimate - upper_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return upper_left


def hdf5_to_dataset(path: Path) -> xr.Dataset:
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {"source_format": "hdf5"}

    with h5py.File(path, "r") as handle:
        def visit(name: str, obj: h5py.Dataset) -> None:
            is_coord = any(alias in name.lower() for alias in (*LON_ALIASES, *LAT_ALIASES, *TIME_ALIASES))
            is_numeric = isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number)
            if isinstance(obj, h5py.Dataset) and (is_numeric or is_coord):
                arrays[name] = np.asarray(obj)
                for attr_key in ("units", "long_name", "standard_name"):
                    if attr_key in obj.attrs:
                        attrs[f"{name}:{attr_key}"] = decode_attr(obj.attrs[attr_key])

        handle.visititems(visit)
        for attr_key, value in handle.attrs.items():
            attrs[attr_key] = decode_attr(value)

    ds = arrays_to_dataset(arrays)
    ds.attrs.update(attrs)
    return ds


def arrays_to_dataset(arrays: Mapping[str, np.ndarray]) -> xr.Dataset:
    lon_name = find_name(arrays, LON_ALIASES)
    lat_name = find_name(arrays, LAT_ALIASES)
    time_name = find_name(arrays, TIME_ALIASES)

    variable_arrays = {
        clean_var_name(name): np.asarray(value)
        for name, value in arrays.items()
        if name not in {lon_name, lat_name, time_name} and np.asarray(value).ndim >= 2
    }
    if not variable_arrays:
        raise ValueError("No numeric grid arrays found.")

    sample = next(iter(variable_arrays.values()))
    lat = np.asarray(arrays[lat_name]).squeeze() if lat_name else np.arange(sample.shape[-2])
    lon = np.asarray(arrays[lon_name]).squeeze() if lon_name else np.arange(sample.shape[-1])
    coords: dict[str, Any] = {"lat": lat, "lon": lon}
    if time_name:
        coords["time"] = stringify_time(np.asarray(arrays[time_name]).squeeze())

    data_vars = {}
    for name, arr in variable_arrays.items():
        dims = infer_array_dims(arr, len(lat), len(lon), len(coords.get("time", [])))
        data_vars[name] = (canonical_dims(dims), orient_array(arr, dims))
    return xr.Dataset(data_vars=data_vars, coords=coords)


def numpy_to_dataset(array: np.ndarray, coords: Mapping[str, Any] | None = None) -> xr.Dataset:
    arr = np.asarray(array, dtype=float)
    coords = dict(coords or {})
    var_name = str(coords.pop("variable", coords.pop("name", "value")))

    if arr.ndim == 2:
        lat = np.asarray(coords.get("lat", np.arange(arr.shape[0])))
        lon = np.asarray(coords.get("lon", np.arange(arr.shape[1])))
        return xr.Dataset({var_name: (("lat", "lon"), arr)}, coords={"lat": lat, "lon": lon})
    if arr.ndim == 3:
        time = coords.get("time", list(range(arr.shape[0])))
        lat = np.asarray(coords.get("lat", np.arange(arr.shape[1])))
        lon = np.asarray(coords.get("lon", np.arange(arr.shape[2])))
        return xr.Dataset(
            {var_name: (("time", "lat", "lon"), arr)},
            coords={"time": stringify_time(time), "lat": lat, "lon": lon},
        )
    if arr.ndim == 4:
        time = coords.get("time", list(range(arr.shape[0])))
        variables = list(coords.get("variables", [f"var_{i}" for i in range(arr.shape[1])]))
        lat = np.asarray(coords.get("lat", np.arange(arr.shape[2])))
        lon = np.asarray(coords.get("lon", np.arange(arr.shape[3])))
        data_vars = {var: (("time", "lat", "lon"), arr[:, idx]) for idx, var in enumerate(variables)}
        return xr.Dataset(data_vars=data_vars, coords={"time": stringify_time(time), "lat": lat, "lon": lon})

    raise ValueError("Numpy arrays must be 2D, 3D, or 4D.")


def standardize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    lon_name = find_coord_name(ds, LON_ALIASES)
    lat_name = find_coord_name(ds, LAT_ALIASES)
    time_name = find_coord_name(ds, TIME_ALIASES)

    if lon_name and lon_name != "lon":
        rename[lon_name] = "lon"
    if lat_name and lat_name != "lat":
        rename[lat_name] = "lat"
    if time_name and time_name != "time":
        rename[time_name] = "time"
    if rename:
        ds = ds.rename(rename)

    if "lat" not in ds.coords and "lat" in ds.dims:
        ds = ds.assign_coords(lat=np.arange(ds.sizes["lat"]))
    if "lon" not in ds.coords and "lon" in ds.dims:
        ds = ds.assign_coords(lon=np.arange(ds.sizes["lon"]))

    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("Could not identify latitude/longitude coordinates.")

    return ds


def build_grid_tensor(
    ds: xr.Dataset,
    *,
    variables: Sequence[str] | None,
    fill_missing: bool,
    fill_method: str,
    fixed_missing_threshold: float,
    source_metadata: Mapping[str, Any],
) -> GridTensor:
    ydim, xdim = spatial_dims(ds)
    selected = select_grid_variables(ds, variables, ydim, xdim)
    if not selected:
        raise ValueError("No numeric grid variables matched the requested selection.")

    arrays = []
    names = []
    units: dict[str, str | None] = {}
    notes: list[str] = []
    for name, da in selected:
        for expanded_name, expanded_da, note in expand_extra_dims(name, da, ydim, xdim):
            arr = canonical_array(expanded_da, ydim, xdim)
            arr = replace_declared_fill_values(arr, expanded_da.attrs)
            arrays.append(arr)
            names.append(expanded_name)
            units[expanded_name] = expanded_da.attrs.get("units")
            if note:
                notes.append(note)

    data = np.stack(arrays, axis=1)
    original_missing = ~np.isfinite(data)
    fixed_missing = original_missing.mean(axis=(0, 1)) >= fixed_missing_threshold

    filled = np.zeros_like(original_missing, dtype=bool)
    output_data = data.copy()
    if fill_missing:
        for tidx in range(output_data.shape[0]):
            for vidx in range(output_data.shape[1]):
                before = ~np.isfinite(output_data[tidx, vidx])
                if before.any():
                    output_data[tidx, vidx] = fill_2d(output_data[tidx, vidx], method=fill_method)
                    filled[tidx, vidx] = before & np.isfinite(output_data[tidx, vidx])

    lat = coord_to_1d(ds["lat"], ydim)
    lon = coord_to_1d(ds["lon"], xdim)
    time = extract_time_values(ds, output_data.shape[0])
    if ds.attrs.get("coordinate_system") == "pixel":
        space_resolution = "1 pixel x 1 pixel"
    else:
        space_resolution = infer_space_resolution(lon, lat)
    resolution = {
        "space": space_resolution,
        "time": infer_time_resolution(time),
    }
    space = {
        "coordinates": {"lon": lon, "lat": lat},
        "bounds": {
            "lon": [float(np.nanmin(lon)), float(np.nanmax(lon))],
            "lat": [float(np.nanmin(lat)), float(np.nanmax(lat))],
        },
        "dimensions": {"lat": int(len(lat)), "lon": int(len(lon))},
        "crs": ds.attrs.get("crs") or ds.attrs.get("spatial_ref") or "EPSG:4326/inferred",
        "coordinate_system": ds.attrs.get("coordinate_system", "geographic_or_projected"),
        "grid_type": ds.attrs.get("grid_type", "spatiotemporal_grid"),
    }
    metadata = {
        "source": dict(source_metadata),
        "dataset_attrs": json_ready(dict(ds.attrs)),
        "grid_type": ds.attrs.get("grid_type", "spatiotemporal_grid"),
        "coordinate_system": ds.attrs.get("coordinate_system", "geographic_or_projected"),
        "fill_missing": fill_missing,
        "fill_method": fill_method,
        "fixed_missing_threshold": fixed_missing_threshold,
        "interpolation_recommendations": interpolation_recommendations(fixed_missing),
        "processing_notes": notes,
    }

    return GridTensor(
        data=output_data,
        space=space,
        time=time,
        variables=names,
        resolution=resolution,
        mask={
            "original_missing": original_missing,
            "filled": filled,
            "fixed_missing": fixed_missing,
            "valid_region": ~fixed_missing,
        },
        metadata=metadata,
        variable_units=units,
    )


def select_grid_variables(
    ds: xr.Dataset,
    variables: Sequence[str] | None,
    ydim: str,
    xdim: str,
) -> list[tuple[str, xr.DataArray]]:
    requested = list(variables or [])
    requested_set = set(requested)
    selected_by_name = {}
    for name, da in ds.data_vars.items():
        if requested_set and name not in requested_set:
            continue
        if ydim not in da.dims or xdim not in da.dims:
            continue
        if not np.issubdtype(da.dtype, np.number):
            continue
        selected_by_name[name] = da

    missing = requested_set - set(selected_by_name)
    if missing:
        raise ValueError(f"Requested variables not found as numeric grid variables: {sorted(missing)}")
    if requested:
        return [(name, selected_by_name[name]) for name in requested]
    return list(selected_by_name.items())


def expand_extra_dims(name: str, da: xr.DataArray, ydim: str, xdim: str) -> Iterable[tuple[str, xr.DataArray, str | None]]:
    extra_dims = [dim for dim in da.dims if dim not in {ydim, xdim, "time"}]
    if not extra_dims:
        yield name, da, None
        return

    for index in np.ndindex(*(da.sizes[dim] for dim in extra_dims)):
        selector = dict(zip(extra_dims, index))
        label_parts = []
        for dim, idx in selector.items():
            coord = da.coords.get(dim)
            value = coord.values[idx] if coord is not None and coord.ndim == 1 else idx
            label_parts.append(f"{dim}={value}")
        expanded_name = f"{name}[{','.join(label_parts)}]"
        yield expanded_name, da.isel(selector), f"Expanded {name} along extra dimensions {extra_dims}."


def canonical_array(da: xr.DataArray, ydim: str, xdim: str) -> np.ndarray:
    if "time" not in da.dims:
        da = da.expand_dims(time=[None])
    da = da.transpose("time", ydim, xdim)
    arr = np.array(da.to_numpy(), dtype=float, copy=True)
    arr[np.abs(arr) > 1.0e30] = np.nan
    return arr


def fill_2d(values: np.ndarray, method: str = "nearest") -> np.ndarray:
    arr = values.astype(float, copy=True)
    missing = ~np.isfinite(arr)
    if not missing.any():
        return arr
    valid = ~missing
    if not valid.any():
        return arr

    if method == "nearest":
        _, indices = ndimage.distance_transform_edt(missing, return_indices=True)
        arr[missing] = arr[tuple(index[missing] for index in indices)]
        return arr

    yy, xx = np.indices(arr.shape)
    points = np.column_stack((yy[valid], xx[valid]))
    targets = np.column_stack((yy[missing], xx[missing]))
    values_valid = arr[valid]

    if method in {"linear", "linear_then_nearest"}:
        filled_values = interpolate.griddata(points, values_valid, targets, method="linear")
        arr[missing] = filled_values
        remaining = ~np.isfinite(arr)
        if method == "linear" or not remaining.any():
            return arr
        missing = remaining
        targets = np.column_stack((yy[missing], xx[missing]))

    if method == "linear_then_nearest":
        nearest = interpolate.griddata(points, values_valid, targets, method="nearest")
        arr[missing] = nearest
        return arr

    raise ValueError("fill_method must be one of: nearest, linear, linear_then_nearest")


def profile_variables(grid_tensor: GridTensor) -> dict[str, dict[str, Any]]:
    profiles = {}
    for idx, name in enumerate(grid_tensor.variables):
        values = grid_tensor.data[:, idx]
        original_missing = grid_tensor.mask["original_missing"][:, idx]
        finite = np.isfinite(values)
        profiles[name] = {
            "unit": grid_tensor.variable_units.get(name),
            "shape": list(values.shape),
            "missing_count_original": int(np.count_nonzero(original_missing)),
            "missing_ratio_original": safe_ratio(np.count_nonzero(original_missing), original_missing.size),
            "filled_count": int(np.count_nonzero(grid_tensor.mask["filled"][:, idx])),
            "min": safe_float(np.nanmin(values)) if finite.any() else None,
            "max": safe_float(np.nanmax(values)) if finite.any() else None,
            "mean": safe_float(np.nanmean(values)) if finite.any() else None,
            "std": safe_float(np.nanstd(values)) if finite.any() else None,
            "suggested_missing_strategy": suggest_variable_strategy(original_missing),
        }
    return profiles


def make_grid_spec(grid_tensor: GridTensor) -> dict[str, Any]:
    lon = grid_tensor.space["coordinates"]["lon"]
    lat = grid_tensor.space["coordinates"]["lat"]
    return {
        "coordinates": {"lon": lon, "lat": lat},
        "resolution": grid_tensor.resolution,
        "dimensions": {
            "lon": int(len(lon)),
            "lat": int(len(lat)),
            "time": int(len(grid_tensor.time)),
            "variables": int(len(grid_tensor.variables)),
        },
        "variables": grid_tensor.variables,
        "bounds": grid_tensor.space["bounds"],
        "crs": grid_tensor.space["crs"],
    }


def summarize_spatiotemporal(
    grid_tensor: GridTensor,
    variable_profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    lon = grid_tensor.space["coordinates"]["lon"]
    lat = grid_tensor.space["coordinates"]["lat"]
    hotspots = {}
    trends = {}
    for vidx, name in enumerate(grid_tensor.variables):
        values = grid_tensor.data[:, vidx]
        if not np.isfinite(values).any():
            hotspots[name] = None
            trends[name] = None
            continue

        finite_count = np.sum(np.isfinite(values), axis=0)
        mean_map = np.divide(
            np.nansum(values, axis=0),
            finite_count,
            out=np.full(values.shape[1:], np.nan, dtype=float),
            where=finite_count > 0,
        )
        yidx, xidx = np.unravel_index(np.nanargmax(mean_map), mean_map.shape)
        hotspots[name] = {
            "lon": safe_float(lon[xidx]),
            "lat": safe_float(lat[yidx]),
            "value": safe_float(mean_map[yidx, xidx]),
        }
        if values.shape[0] > 1:
            first = np.nanmean(values[0])
            last = np.nanmean(values[-1])
            trends[name] = {
                "first_mean": safe_float(first),
                "last_mean": safe_float(last),
                "delta": safe_float(last - first),
            }
        else:
            trends[name] = "single_time_step"

    original_missing = grid_tensor.mask["original_missing"]
    fixed_missing = grid_tensor.mask["fixed_missing"]
    return {
        "time_coverage": {
            "start": grid_tensor.time[0] if grid_tensor.time else None,
            "end": grid_tensor.time[-1] if grid_tensor.time else None,
            "count": len(grid_tensor.time),
            "resolution": grid_tensor.resolution["time"],
        },
        "space_coverage": grid_tensor.space["bounds"],
        "missing_summary": {
            "original_missing_ratio": safe_ratio(np.count_nonzero(original_missing), original_missing.size),
            "fixed_missing_cell_ratio": safe_ratio(np.count_nonzero(fixed_missing), fixed_missing.size),
            "filled_count": int(np.count_nonzero(grid_tensor.mask["filled"])),
            "recommendations": grid_tensor.metadata["interpolation_recommendations"],
        },
        "hotspots_by_variable": hotspots,
        "temporal_trends": trends,
        "variable_profile_keys": list(variable_profiles.keys()),
    }


def spatial_dims(ds: xr.Dataset) -> tuple[str, str]:
    lat = ds["lat"]
    lon = ds["lon"]
    if lat.ndim == 1 and lon.ndim == 1:
        return lat.dims[0], lon.dims[0]
    shared = [dim for dim in lat.dims if dim in lon.dims]
    if len(shared) >= 2:
        return shared[-2], shared[-1]
    raise ValueError("Latitude/longitude coordinates must be 1D or share 2D grid dimensions.")


def coord_to_1d(coord: xr.DataArray, dim: str) -> np.ndarray:
    if coord.ndim == 1:
        return coord.to_numpy()
    axis = coord.dims.index(dim)
    indexer = [0] * coord.ndim
    indexer[axis] = slice(None)
    return coord.to_numpy()[tuple(indexer)]


def replace_declared_fill_values(arr: np.ndarray, attrs: Mapping[str, Any]) -> np.ndarray:
    out = arr.astype(float, copy=True)
    for key in FILL_VALUE_ATTRS:
        if key not in attrs:
            continue
        values = np.atleast_1d(attrs[key])
        for value in values:
            try:
                out[out == float(value)] = np.nan
            except (TypeError, ValueError):
                continue
    return out


def infer_space_resolution(lon: np.ndarray, lat: np.ndarray) -> str:
    lon_res = median_step(lon)
    lat_res = median_step(lat)
    if lon_res is None or lat_res is None:
        return "unknown"
    return f"{lon_res:g} degree lon x {lat_res:g} degree lat"


def infer_time_resolution(time: Sequence[str | None]) -> str:
    if len(time) <= 1:
        return "single time"
    parsed = pd.to_datetime([t for t in time if t is not None], errors="coerce")
    parsed = parsed[~pd.isna(parsed)]
    if len(parsed) <= 1:
        return "unknown"
    deltas = pd.Series(parsed).diff().dropna().dt.total_seconds().to_numpy()
    median_seconds = float(np.median(deltas))
    if median_seconds % 86400 == 0:
        days = int(median_seconds // 86400)
        return f"{days} day" if days == 1 else f"{days} days"
    if median_seconds % 3600 == 0:
        hours = int(median_seconds // 3600)
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    return f"{median_seconds:g} seconds"


def median_step(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    diffs = np.diff(arr)
    diffs = np.abs(diffs[np.isfinite(diffs) & (np.abs(diffs) > 0)])
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def interpolation_recommendations(fixed_missing: np.ndarray) -> list[str]:
    fixed_ratio = safe_ratio(np.count_nonzero(fixed_missing), fixed_missing.size)
    recommendations = [
        "Use nearest-neighbor fill for fixed outer-domain gaps so boundary cells do not create artificial gradients.",
        "Use linear interpolation for sparse internal missing cells in smooth meteorological fields.",
        "Keep fixed_missing_mask as a downstream validity mask even when filled values are used for model input.",
    ]
    if fixed_ratio > 0.2:
        recommendations.append(
            "A large fixed missing region was detected; report metrics both with and without the valid_region mask."
        )
    return recommendations


def suggest_variable_strategy(original_missing: np.ndarray) -> str:
    ratio = safe_ratio(np.count_nonzero(original_missing), original_missing.size)
    fixed_ratio = safe_ratio(
        np.count_nonzero(original_missing.mean(axis=0) >= 0.8),
        original_missing.shape[-2] * original_missing.shape[-1],
    )
    if fixed_ratio > 0.05:
        return "nearest for fixed outside-domain cells; linear_then_nearest for internal gaps"
    if ratio > 0:
        return "linear interpolation for sparse gaps; nearest fallback on edges"
    return "no fill required"


def extract_time_values(ds: xr.Dataset, expected_count: int) -> list[str | None]:
    if "time" not in ds.coords:
        return [None] * expected_count
    time = ds["time"]
    if time.ndim == 0:
        return [format_time_value(time.values)] * expected_count
    return [format_time_value(value) for value in time.values[:expected_count]]


def format_time_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        return timestamp.isoformat()
    except Exception:
        return str(value)


def stringify_time(values: Any) -> list[str]:
    return [format_time_value(value) or str(value) for value in np.atleast_1d(values)]


def infer_time_from_filename(path: Path) -> str | None:
    stem = path.stem
    for fmt_len, fmt in ((8, "%Y%m%d"), (10, "%Y-%m-%d")):
        for start in range(0, max(1, len(stem) - fmt_len + 1)):
            token = stem[start : start + fmt_len]
            try:
                return pd.to_datetime(token, format=fmt).isoformat()
            except ValueError:
                continue
    return None


def find_coord_name(ds: xr.Dataset, aliases: Sequence[str]) -> str | None:
    candidates = list(ds.coords) + list(ds.dims)
    return find_column(candidates, aliases)


def find_column(columns: Iterable[Any], aliases: Sequence[str]) -> str | None:
    alias_set = {alias.lower() for alias in aliases}
    for col in columns:
        lower = str(col).lower()
        if lower in alias_set:
            return str(col)
    for col in columns:
        lower = str(col).lower()
        if any(alias in lower for alias in alias_set):
            return str(col)
    return None


def find_name(mapping: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    return find_column(mapping.keys(), aliases)


def clean_var_name(name: str) -> str:
    return Path(name).name.replace(" ", "_")


def infer_array_dims(arr: np.ndarray, lat_len: int, lon_len: int, time_len: int) -> tuple[str, ...]:
    if arr.ndim == 2:
        return ("lat", "lon")
    if arr.ndim == 3:
        shape = arr.shape
        if shape[-2:] == (lat_len, lon_len):
            return ("time", "lat", "lon")
        if shape[:2] == (lat_len, lon_len):
            return ("lat", "lon", "time")
        if time_len and shape[0] == time_len:
            return ("time", "lat", "lon")
    raise ValueError(f"Could not infer dimensions for array with shape {arr.shape}")


def canonical_dims(dims: tuple[str, ...]) -> tuple[str, ...]:
    if dims == ("lat", "lon", "time"):
        return ("time", "lat", "lon")
    return dims


def orient_array(arr: np.ndarray, dims: tuple[str, ...]) -> np.ndarray:
    if dims == ("lat", "lon", "time"):
        return np.transpose(arr, (2, 0, 1))
    return arr


def apply_coord_overrides(ds: xr.Dataset, coords: Mapping[str, Any]) -> xr.Dataset:
    assignable = {key: value for key, value in coords.items() if key in {"lat", "lon", "time"}}
    return ds.assign_coords(assignable) if assignable else ds


def decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def safe_ratio(part: int, total: int) -> float:
    return float(part / total) if total else 0.0


def cli_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    grid_spec = dict(result["grid_spec"])
    grid_spec["coordinates"] = {
        "lon": {
            "size": int(len(grid_spec["coordinates"]["lon"])),
            "min": safe_float(np.nanmin(grid_spec["coordinates"]["lon"])),
            "max": safe_float(np.nanmax(grid_spec["coordinates"]["lon"])),
        },
        "lat": {
            "size": int(len(grid_spec["coordinates"]["lat"])),
            "min": safe_float(np.nanmin(grid_spec["coordinates"]["lat"])),
            "max": safe_float(np.nanmax(grid_spec["coordinates"]["lat"])),
        },
    }
    return {
        "grid_spec": grid_spec,
        "grid_tensor": result["grid_tensor"].summary(),
        "variable_profiles": result["variable_profiles"],
        "spatiotemporal_summary": result["spatiotemporal_summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and standardize grid data into GridTensor.")
    parser.add_argument("source", help="Input file, directory, or supported grid data path.")
    parser.add_argument("--variables", nargs="*", default=None, help="Optional variables to keep.")
    parser.add_argument(
        "--fill-method",
        default="nearest",
        choices=("nearest", "linear", "linear_then_nearest"),
        help="Missing value fill method.",
    )
    parser.add_argument("--no-fill", action="store_true", help="Keep missing values unfilled.")
    args = parser.parse_args()

    result = load_grid_data(
        args.source,
        variables=args.variables,
        fill_missing=not args.no_fill,
        fill_method=args.fill_method,
    )
    print(json.dumps(cli_summary(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
