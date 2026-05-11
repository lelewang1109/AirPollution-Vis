# Data Instructions

The `data/` folder contains large NetCDF files for the demo dataset. These files are excluded from GitHub because the dataset size is large and not suitable for source control.

## Expected layout

The project expects raw data files under:

```text
data/2000/
```

## Setup

1. Obtain the NetCDF files from your local copy or external dataset source.
2. Place them in `data/2000/`.
3. Run the server with `python gridvis_server.py`.

## Notes

- The application uses `data/2000/*.nc` files as the main input.
- Keep the `data/` directory on your local machine if the dataset is too large for GitHub.
