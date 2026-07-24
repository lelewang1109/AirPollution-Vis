# AirPollution-Vis

A local visual analytics demo for gridded environmental data. The Python backend serves the demo UI and exposes APIs consumed by the frontend.

## Repository layout

- `gridvis_server.py` — root launcher for the application.
- `src/gridvis_app/` — backend server and API implementation.
- `frontend/final/` — final presentation interface.
- `modules/` — reusable pipeline modules for data access, grid representation, LLM planning, tool execution, and output explanation.
- `assets/` — shared static assets such as `china.json`.
- `data/` — raw NetCDF input dataset (large, excluded from GitHub).
- `gridvis-venv/` — local Python virtual environment (ignored by Git).

## Quick start

1. Create or activate a virtual environment:

```bash
python3 -m venv gridvis-venv
source gridvis-venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Optional: install the package in editable mode:

```bash
pip install -e .
```

Then run the server with either:

```bash
python gridvis_server.py
```

or:

```bash
gridvis
```

4. Open the UI in your browser:

```text
http://127.0.0.1:8787/
```

If port `8787` is busy, the server will automatically try the next available port.

## Data

The NetCDF dataset is not included in this repository.

Data source: [Science Data Bank (SciDB), version V3](https://www.scidb.cn/preview?dataSetId=f782b19807ce4b1299563e6dfaf67a91&version=V3).

Download the required files from the source above and place them in `data/2000/`. See `DATA.md` for the expected local layout.

## Health checks

```bash
curl http://127.0.0.1:8787/api/health
curl http://127.0.0.1:8787/api/catalog
```

## Notes

- Keep `gridvis-venv/` outside Git; this repo tracks source code only.
- Use `frontend/final/index.html` for the published demo interface.
