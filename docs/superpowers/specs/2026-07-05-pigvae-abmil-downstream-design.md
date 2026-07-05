# PIGVAE → Gated-ABMIL downstream evaluation — design

- **Date:** 2026-07-05
- **Status:** approved (brainstorm), pending implementation plan
- **Repo:** `IMMUVIS-PIGVAE` (branch `imc-pigvae-film-multi`)
- **Related:** `[[exp-2026-06-29-vae16-ae-vae-spectrum]]` (the substrate whose `z` we now test on transfer — the thesis's deciding metric)

## 1. Purpose

Test whether the frozen PIGVAE latent transfers to a clinical downstream task — the deciding
metric the reconstruction sweeps could not rank. Concretely: use the frozen PIGVAE encoder to turn
IMC crops into a single per-crop representation (`z_global`, the CLS token), then train a gated-ABMIL
classifier over image bags of those crops for clinical features, under 10-fold cross-validation.

This ports the existing gated-ABMIL pipeline (`/home/tnocon/mil/gated_abmil`) into this repo and
standardises it to the repo's Hydra conventions, adding one new stage: **PIGVAE's CLS replaces the
mean-pool** as the per-crop vector.

## 2. Scope

**In scope**
- A new **encode** stage: frozen PIGVAE + saved fitted PCA → `z_global` per crop → memmap `.npy` +
  metadata CSV, in the schema the ABMIL side already consumes.
- A ported, Hydra/Lightning **ABMIL** stage: `GatedABMIL` + 10-fold StratifiedKFold per clinical
  feature, emitting the same metrics CSVs as the existing pipeline.
- GPU verification of both stages on szary.

**Out of scope**
- Building the crop→image→label meta CSVs (the existing `logistic_regression` step already produces
  these; we consume them and re-point `embeddings_file`).
- Training/finetuning PIGVAE (encoder is frozen).
- Reading raw IMC tiffs (encode consumes the existing per-crop 16×16×768 arrays in place).
- Turning the permuter on / invariance training.

## 3. Data flow

```
existing per-crop IMC arrays (16×16×768; what immuvis mean-pools)
      │  referenced by train/test meta CSVs (img_path, coords0/1, dataset, feature, feature_value)
      ▼
[ENCODE]  PIGVAE fitted PCA 768→128  →  16×16 grid graph (DenseGraphBatch)
          →  frozen PIGVAE.encode(sample=False)  →  z_global [D≈512]
      │  stream rows → memmap .npy   +   metadata CSV  (embedding_idx aligned, labels preserved)
      ▼
[ABMIL]   build_image_bags (bag = img_path, instances = crop z_globals)
          →  10-fold StratifiedKFold GatedABMIL per (dataset, clinical feature)
      │  → cv_folds.csv, cv_summary.csv, results.csv  (drop-in comparable to other models)
```

The two stages are **decoupled by files on disk**: encode once (6M train crops), sweep ABMIL cheaply
many times.

## 4. Components

New package `src/downstream/`, two Hydra entry-point scripts.

### 4.1 `scripts/encode_mil_embeddings.py` + `src/downstream/encode.py`
- **Does:** load a PIGVAE checkpoint and the saved `PCALayer`
  (`pca_model_{num_pca_components}_center_crop_{center_crop_size}.pkl`), `eval()` + `no_grad()`;
  iterate crops **per (dataset, split) shard** (cords/danenberg × train/test) from the meta CSVs →
  the existing 16×16×768 arrays; apply PCA(768→128) → build `DenseGraphBatch` →
  `graph_ae.encode(sample=False)` → `z_global`; **stream** rows into a memmap `.npy` and write a
  metadata CSV with `embedding_idx` aligned 1:1 to memmap rows.
- **Interface:** input = meta CSV dir + crop-array source; output = `{dataset}_{split}_embeddings.npy`
  + `{dataset}_{split}_metadata.csv` pairs (the layout `concat_metadata` already expects).
- **Depends on:** `graph_ae.encode`, `PCALayer`, `DenseGraphBatch`, a PIGVAE checkpoint. GPU-batched
  (batch size configurable); resumable (skip a shard whose outputs already exist).
- **Frozen/deterministic:** `sample=False` → `z_global = layer_norm(graph_emb)`; PCA transform-only.

### 4.2 `src/downstream/abmil/model.py`
- `GatedABMIL` nn.Module ported ~verbatim (it is already clean: gated attention V/U/W, masked
  softmax, bag pooling, classifier head).
- Thin `pl.LightningModule` wrapper: `training_step`/`validation_step`/`configure_optimizers`;
  BCEWithLogits (2-class) / CrossEntropy (multiclass); Lightning `EarlyStopping` + `ModelCheckpoint`.

### 4.3 `src/downstream/abmil/data.py`
- `MILDataset` + `collate_fn`: pad bags to `max_len`, build padding `mask` (B×S), return
  `(bags B×S×D, masks B×S, labels B)`.
- `build_image_bags(meta_df, class_to_idx)`: group by `img_path`, one bag per image, instances =
  that image's crop `z_global` rows from the memmap; carry `feature_value` → label. Drops NaN labels.

### 4.4 `src/downstream/abmil/cv.py`
- `balance_meta_df` (min class freq), `drop_nan_labels`, StratifiedKFold (folds capped by min class
  count), per-fold: build dataloaders + fresh LightningModule + `Trainer.fit` → OOF predictions.
- Metrics: accuracy, macro-F1, AUC (binary via sigmoid, multiclass via softmax OVR). Emits
  `cv_folds`, `cv_summary`, `results` CSVs matching the current pipeline's columns.

### 4.5 Configs
- `configs/downstream/encode.yaml`: `ckpt_path`, `pca_pkl`, `meta_in_dir`, `emb_out_dir`,
  `batch_size`, `device`, `shards` (dataset×split). Composes `paths` (`szary.yaml`).
- `configs/downstream/abmil.yaml`: `meta_dir`, `features` (Grade, Relapse, DX.name, ERStatus,
  ERBB2_pos, PAM50), `datasets` (cords, danenberg), `num_folds: 10`, `hidden_dim`, `num_heads`,
  `num_epochs`, `patience`, `lr`, `zscore ∈ {none, global, cv_train}`, `results_dir`.

## 5. Correctness / leakage guardrails

1. **Same frozen encoder + same fitted PCA** for all splits, **transform-only** (never refit on
   downstream data) — keeps the representation space consistent (this is *required*, not a leak).
2. **Label alignment** asserted end-to-end: memmap row ↔ metadata `embedding_idx` ↔ `img_path` ↔
   `feature_value`.
3. **Normalization fit on train fold only** — keep `zscore='cv_train'`; `'global'` is flagged leaky
   (fits mean/std on pooled train+val), as the current code already documents.
4. **All crops of an image stay in one split** — guaranteed by grouping/splitting at `img_path` level.
5. **Documented caveat:** if ABMIL-val images overlap PIGVAE's self-supervised pretraining set, the
   val features are mildly optimistic (label-free overlap). State it; prefer PIGVAE-held-out images
   for the probe val where feasible.

## 6. Error handling

- Missing embedding/meta file → clear error naming the shard.
- NaN labels dropped; classes below `min_class_freq` dropped (`balance_meta_df`).
- Folds capped to `min(num_folds, min_class_count, n_bags)`; fall back to a single stratified split
  when a class has <2 samples (existing logic).
- Memmap dtype/shape mismatch → assert on write and on read.
- Checkpoint / PCA path resolution via `paths` group (szary vs login).
- GPU OOM → configurable encode/ABMIL batch size; CPU fallback with a warning.

## 7. Testing

- **Unit (encode):** on a synthetic tiny meta table + mock crop arrays — memmap has correct shape,
  metadata `embedding_idx` aligns 1:1, `z_global` is deterministic at `sample=False`, PCA is
  transform-only (no refit).
- **Unit (abmil):** `build_image_bags` groups by `img_path` with aligned labels; `collate_fn` pads +
  masks correctly; StratifiedKFold split is reproducible (seed 42); AUC/F1 correct on toy inputs;
  `GatedABMIL.forward` shape in/out.
- **Integration:** tiny end-to-end (handful of synthetic crops → encode → 1-fold ABMIL) runs clean.
- **GPU smoke:** one real-data run of each stage on szary confirms the CUDA path (resolves the
  standing "GPU problem" by running, not asserting).

## 8. Open questions / risks

- **Input crop-array format — RESOLVED (verified on szary 2026-07-05).** Encode reads the raw cords
  IMC patches at `/raid_encrypted/immucan/embeddings/tnocon/data/IMC/cords/{train,test}.h5`, NOT the
  ImmuVis-UN-361 embeddings folder. HDF5 layout: `embeddings` (N, 768, 16, 16) float32, `paths` (N,)
  → img_path, `positions` (N, 4) → coords, `metadata` (N, 8, 3). PCA artifact
  `pca_model_128_center_crop_16.pkl` (sklearn PCA, 768→128) + `imc_statistics_128_center_crop_16.pt`
  (mean/std over PCA-128) apply directly. test.h5 = 10197, train.h5 = 40843 — these are PIGVAE's own
  train/val split (so the §5 pretraining-overlap caveat is live: ABMIL-val patches from train.h5 were
  seen self-supervised by the encoder). Encode transform: patch (768,16,16) → 256 nodes × 768 →
  PCA→128 → normalize(mean,std) → DenseGraphBatch → encode(sample=False) → z_global.
- z_global width is read from the data at ABMIL time (`embedding_dim = bags[0].shape[1]`), so no
  hardcoding; confirm it is the intended CLS (`layer_norm(graph_emb)`, ~512-d) and not `z_nodes`.
- Encode throughput at 6M crops: needs GPU batching + streaming memmap; sharding by (dataset, split)
  gives resume + parallelism.
- Which PIGVAE checkpoint to encode with (the AE-ward s0.001 vs balanced s0.01 substrate) is a
  downstream experiment, not a design decision — the config takes `ckpt_path`, so both are runnable.

## 9. File layout summary

```
scripts/encode_mil_embeddings.py
scripts/run_abmil.py
src/downstream/__init__.py
src/downstream/encode.py
src/downstream/abmil/__init__.py
src/downstream/abmil/model.py
src/downstream/abmil/data.py
src/downstream/abmil/cv.py
configs/downstream/encode.yaml
configs/downstream/abmil.yaml
tests/test_downstream_encode.py
tests/test_downstream_abmil.py
```
