# Data Adapter

Step 1 converts supported grid data sources into a canonical `GridTensor`.

Supported inputs:

- NetCDF: `.nc`, `.netcdf`
- CSV table or numeric matrix: `.csv`
- GeoTIFF: `.tif`, `.tiff`
- PNG image grids: `.png`
- HDF5: `.h5`, `.hdf5`
- Numpy: `.npy`, `.npz`, or an in-memory `numpy.ndarray`
- In-memory `xarray.Dataset` / `xarray.DataArray`

The canonical tensor uses this shape:

```text
(time, variable, lat, lon)
```

Basic usage from the repository root:

```bash
./gridvis-venv/bin/python modules/01_data_adapter/adapter.py data/2000/20000101.nc
```

PNG images are treated as generic pixel grids. RGB images are converted into
`intensity`, `red`, `green`, and `blue` variables using pixel coordinates:

```bash
./gridvis-venv/bin/python modules/01_data_adapter/adapter.py data/image.png
```

Python usage:

```python
import sys

sys.path.insert(0, "modules/01_data_adapter")
from adapter import load_grid_data

result = load_grid_data("data/2000/20000101.nc")

grid_spec = result["grid_spec"]
grid_tensor = result["grid_tensor"]
profiles = result["variable_profiles"]
summary = result["spatiotemporal_summary"]

print(grid_tensor.data.shape)
print(grid_tensor.mask["fixed_missing"].shape)
```

Missing values are handled in two layers:

- `original_missing` records all missing cells before any fill.
- `fixed_missing` marks cells that are repeatedly missing across variables/time, such as outside-domain regions.
- `filled` records cells filled by the selected interpolation method.

Default filling uses nearest-neighbor interpolation because it is stable at fixed domain boundaries. Use `--fill-method linear_then_nearest` for smoother internal gaps with nearest fallback at edges.
