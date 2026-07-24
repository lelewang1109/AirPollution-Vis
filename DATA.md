# Data Instructions

The NetCDF files required by this project are not stored in the GitHub repository.

## Data source

[Science Data Bank (SciDB), version V3](https://www.scidb.cn/preview?dataSetId=f782b19807ce4b1299563e6dfaf67a91&version=V3)

## Expected layout

After downloading the data, place the required NetCDF files under:

```text
data/2000/
```

## Setup

1. Download the dataset from the Science Data Bank link above.
2. Place them in `data/2000/`.
3. Run the server with `python gridvis_server.py`.

## Notes

- The application uses `data/2000/*.nc` files as the main input.
- The entire `data/` directory is ignored by Git and must remain local.
