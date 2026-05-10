# GridVis-LLM

GridVis-LLM is a local visual analytics demo for gridded environmental data. The final demo UI is served by a small Python HTTP server; the same server also exposes the analysis APIs consumed by the frontend.

## Run

```bash
cd /Users/lele/Desktop/GridVis-LLM
./gridvis-venv/bin/python gridvis_server.py
```

Open the URL printed by the terminal. By default it is:

```text
http://127.0.0.1:8787/
```

If that port is busy, the server automatically tries the next ports. You can also force a port:

```bash
GRIDVIS_PORT=8788 ./gridvis-venv/bin/python gridvis_server.py
```

## Main Paths

- `gridvis_server.py`: compatibility launcher from the project root.
- `src/gridvis_app/server.py`: actual backend server and API implementation.
- `frontend/final/`: final presentation interface.
- `modules/`: reusable pipeline modules for data access, representation, LLM planning, tool execution, and result explanation.
- `legacy_frontends/`: earlier per-stage frontend prototypes kept for reference.
- `assets/`: shared static assets such as `china.json`.
- `data/2000/`: NetCDF input data.
- `docs/`: runbooks, reference notes, diagrams, and media.

## API Checks

```bash
curl http://127.0.0.1:8787/api/health
curl http://127.0.0.1:8787/api/catalog
```

The final UI is available at `/frontend/final/index.html`. The old `/03_llm_core_frontend/index.html` path is still supported as a compatibility alias.
