# Design: IMC inference API on Modal

Status: draft for review · 2026-11-11

## Intent

A private HTTP API where the author sends a raw IMC image of the cords panel and gets back the
clinical prediction plus the explanations the thesis pipeline can produce. A UI will be built
separately on top of this API, so the API's JSON is the product. The ONNX stub in
`src/deploy/modal/app.py` is replaced.

The pipeline is a chain of four trained parts:

```
raw TIFF [43, H, W]
  -> preprocess (arcsinh(x/5) -> Butterworth low-pass on the whole image -> clip(0, 4)/4)
  -> 128 x 128 tiles (non-overlapping, row-major, remainder at right/bottom dropped)
  -> ImmuVis encoder            -> [n, 768, 16, 16]
  -> PCA-128 per node           -> [n, 256, 128]
  -> PIGVAE encoder (frozen)    -> z_global [n, 512]
  -> z-score (fold statistics) -> gated ABMIL head (DX.name) -> logit, attention [n]
  -> Integrated Gradients over the bag -> importance [n]
```

## Goals

- `POST /predict`: TIFF in, prediction and per-crop explanations out.
- `POST /embed`: TIFF in, per-crop `z_global` out (stops after PIGVAE).
- `GET /info`, `GET /healthz`.
- Identical numbers to the thesis pipeline for the same image (parity check, below).
- Only the author can call it; the weights never leave the Modal workspace.
- Costs nothing when idle.

## Non-goals

- The UI.
- The ONNX version. A later, separate learning exercise (probably PIGVAE only, inference only).
- Panels other than cords. The PCA, PIGVAE and ABMIL heads were trained on cords only.
- Labels other than `DX.name` for now (the bundle format should not prevent adding Grade,
  Relapse and Stage later).
- Ensembling the 10 folds.
- Storing uploaded images.

## Anti-goals (a failure even if it runs)

- Preprocessing or tiling that differs silently from training, so predictions are quietly wrong.
- An endpoint anyone with the URL can call.
- A deployment that depends on files in the author's home folder or on the raid paths.

## Decisions

### One Modal app, PyTorch

- One GPU class loads every model once at container start (`@modal.enter`); requests run the whole
  chain in memory. Chosen over three chained services because attribution needs gradients through
  the chain and the models are small.
- PyTorch, not ONNX: Integrated Gradients needs autograd, and ImmuVis builds per-marker weights at
  runtime, which exports poorly.
- GPU: a small one (L4). `max_containers=1`, `scaledown_window` a few minutes, `min_containers=0`.

### Code layout

- `src/deploy/pipeline.py`: the model chain as plain PyTorch with no Modal import, so it runs and is
  tested locally. Loads from a bundle directory (below).
- `src/deploy/modal/app.py`: Modal image, Volume, GPU class and the FastAPI app.
- Code shared with the thesis figures moves from `scripts/thesis_figures/figlib/` into `src/`, and
  figlib imports it from there:
  - the ImmuVis loader and cords channel ids (`figlib/immuvis.py`),
  - preprocessing and tiling (`figlib/imc.py`: `preprocess`, `crops`),
  - PCA projection (`figlib/chain.py`: `project`).
- ImmuVis code comes from the `external/multiplex-image-model` submodule (branch `abmil`), which
  is added to the container image.
- `pyproject.toml`: a `deploy` dependency group with `modal` and `fastapi` (the container image
  installs its own pinned list).

### Model bundle

`scripts/deploy/export_bundle.py` runs on szary and writes one self-contained directory, which is
uploaded to a Modal Volume. Nothing in the container reads Hydra configs, W&B or raid.

| File | Content |
|------|---------|
| `immuvis.pth`, `immuvis.yaml` | ImmuVis-475-768-ma1-refactored weights and config |
| `pca.pt` | PCA `components` [128, 768] and `mean` [768] as tensors (no sklearn at runtime) |
| `pigvae.pt` | `graph_ae` state dict + resolved `graph_ae` hparams (experiment `pigvae`, run `81zasevr`) |
| `abmil_DX.name.pt` | head state dict + hparams, z-score `mu`/`sd`, class names, fold index, fold validation AUROC |
| `markers.json` | cords marker names in channel order and their ImmuVis token ids |
| `manifest.json` | source paths, checkpoint hashes, git commit, export date |

Fold choice: the folds are rebuilt with the same code as the CV (`StratifiedKFold(seed=42)`,
same filtering, as `figlib/abmil.py` does); the script computes each fold's validation AUROC,
keeps the best fold by default (`--fold` overrides), and recomputes that fold's z-score
statistics from its training bags. `/info` reports the fold, so the slight optimism of a
best-fold pick is visible.

Upload: `modal volume put` from a machine with Modal credentials. The bundle is written to a path
the login node can read if szary has no internet access.

### Parity check (part of export)

The export script runs `src/deploy/pipeline.py` on the bundle for a few training images and
checks, per image:

- ImmuVis maps against the stored h5 embeddings: correlation > 0.9999.
- The bag logit against the CV out-of-fold logit of that image for the chosen fold's validation
  images: close to 1e-4.

Export fails if any check fails.

### API

Auth: `@modal.asgi_app(requires_proxy_auth=True)`. Callers send `Modal-Key` and `Modal-Secret`
headers (a proxy-auth token created in the Modal dashboard); requests without them are rejected by
Modal before reaching the container, so they do not start a GPU.

`POST /predict` (multipart `file`: a TIFF `[43, H, W]` of raw ion counts in cords marker order;
query `include_embeddings: bool = false`):

```json
{
  "label": "DX.name",
  "prediction": "Squamous cell carcinoma",
  "probabilities": {"Adenocarcinoma": 0.12, "Squamous cell carcinoma": 0.88},
  "logit": 1.99,
  "image": {"height": 753, "width": 711, "tile_size": 128, "n_crops": 30},
  "crops": [
    {"row": 0, "col": 0, "attention": 0.031, "ig_importance": 0.42, "z_global": null}
  ],
  "model": {"fold": 3, "fold_val_auroc": 0.91, "bundle": "<manifest hash>"},
  "timing_ms": {"preprocess": 0, "immuvis": 0, "pigvae": 0, "abmil": 0, "ig": 0}
}
```

`row`/`col` are the top-left pixel of each crop, in the same row-major order as training.

`POST /embed`: same input, returns `image` and `crops` with `z_global` only.

`GET /info`: label, classes, fold, expected marker list, tile size, bundle manifest.

Input validation (HTTP 422): not a TIFF, not 3-D, channel count ≠ 43, smaller than 128 px in
either dimension, larger than a size cap (e.g. 4096 px per side).

### Client

`scripts/deploy/query_modal.py` is rewritten: send a TIFF to `/predict` with the proxy-auth headers
read from environment variables and print a summary.

### Tests

- Unit (CPU, no data): tiling coordinates and order, preprocessing shape/range, input validation,
  the pipeline end to end on a tiny randomly initialised bundle, response schema.
- `RunIf(imc_data=True)`: bundle parity on real data (same checks as the export script).

## Left to the builder

- Exact container package pins (match `uv.lock` for torch, timm, scikit-image).
- IG steps and baseline (default: zero bag, as in `explain_abmil.py`) and whether IG is optional
  per request.
- Volume and app names; the size cap value.
- Whether `figlib` keeps thin wrappers or imports directly from `src`.
