# Grid Representation

Step 2 extracts multiscale features from the Step 1 `GridTensor` representation.

Feature groups:

- `global_statistics`: min, max, mean, std, missing ratio, and variable correlations.
- `spatial_features`: hotspot components, hotspot locations, gradients, anisotropy, connected domains, and boundary change.
- `temporal_features`: trend, change points, periodicity, volatility, and per-variable time summaries.
- `image_features`: image-grid edge density, local texture, local entropy, HSV/colorfulness summaries, channel summaries, and intensity histogram.
- `transformer_features`: patch tokens, 2D positional encoding, lightweight multi-head self-attention, and adaptive saliency.
- `embeddings`: deterministic low-dimensional vectors for blocks, time slices, and variable relations.
- `semantic_summary`: short structured facts intended for LLM task understanding.

Run from the repository root:

```bash
./gridvis-venv/bin/python modules/02_grid_representation/feature_extractor.py data/2000/20000101.nc --variables PM2.5 temp rhum
```

For image grids:

```bash
./gridvis-venv/bin/python modules/02_grid_representation/feature_extractor.py data/image.png
```

Python usage:

```python
import sys

sys.path.insert(0, "modules/02_grid_representation")
from feature_extractor import extract_grid_features

features = extract_grid_features(
    "data/2000/20000101.nc",
    variables=["PM2.5", "temp", "rhum"],
    block_shape=(32, 32),
    embedding_dim=8,
)

print(features["global_statistics"]["correlation"])
print(features["spatial_features"]["hotspot_count"])
print(features["semantic_summary"]["llm_facts"])
```

The extractor accepts either:

- a `GridTensor`
- the full Step 1 result dictionary containing `grid_tensor`
- a file path, directory, or file sequence accepted by `01_data_adapter.load_grid_data`

The embedding implementation uses deterministic SVD projection through NumPy, so no extra machine-learning dependency is required.

The Transformer representation is implemented locally in
`transformer_grid_encoder.py`. It is a dependency-free baseline that mirrors a
ViT-style flow: grid patches become tokens, 2D position encodings are added, a
multi-head self-attention block computes patch relationships, and the output
includes a saliency ranking. Saliency is not pure attention; it fuses visual
distinctiveness (gradient, texture, dynamic range, entropy, colorfulness),
attention centrality, and token feature energy, then smooths the patch map for
display. This schema can later be backed by pretrained ViT, CLIP, or BLIP-style
encoders without changing downstream steps.
