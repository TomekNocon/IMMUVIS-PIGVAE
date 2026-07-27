# Pulling wandb History Cleanly

`get_run_history_tool` is the only way to get per-step curves, but two mechanics
will silently give you a *coarse, misleading* picture if you ignore them. Both are
limits of the **tool response**, not of how much you can read.

## The two mechanics

1. **Response size cap (~75k chars).** If a single response would exceed it, the
   tool truncates and dumps to a file. Asking for *more* `samples` doesn't help —
   the cap bites first, so you get the same ~75k either way.
2. **Downsampling.** `samples` (default 500) returns that many evenly-spaced rows,
   not every row. A 40-epoch run logs ~1700 rows; you get a thinned subset.

**Why this corrupts per-epoch metrics:** `val_mse_loss` is logged **once per
epoch** (~40 rows among ~1700). Pull *all* keys at once and each row is fat
(~40 metrics), so only ~84 rows fit under the cap — and only ~4 of those land on
an epoch-boundary val row. Result: a 4-point "val curve" that hides the actual
overfit onset. This is exactly the trap to avoid.

## The method

**For sparse, per-epoch metrics (the ones that decide the verdict) — fetch ONE key
at a time.** A single-key request is tiny, fits under the cap, and returns *every*
epoch in one call:

```
get_run_history_tool(..., run_id, keys=["val_mse_loss"], samples=200)
→ all 40 epochs, clean.
```

Do the same per sparse key you need: `val_mse_loss`, `val_kld_loss`,
`latent/std_mean`, `latent/mu_std`, `latent/kld_per_dim_median`,
`latent/active_dims`, `latent/kld_alpha`. (Mixing a sparse key with dense
per-step keys in one `keys=[...]` call returns **empty** — a tool quirk; keep
single-key calls single.)

**For dense, per-step metrics** (`mse_loss`, `kld_loss`, actmon `max_abs`), the
full-run downsample is fine for the *shape*. If you need a region at full
resolution (the α-ramp, or the back-half collapse), pull a **step window**
(`min_step`/`max_step`) so the row budget covers fewer steps densely. Stitch a few
windows for a dense full-run view of a dense key.

**Parsing.** Responses save to a file as `{"result": "<json-string>"}`. Parse with
`jq -r '.result | fromjson | .rows | ...'`. Per-epoch keys and per-step keys live
on different rows, so `select(.<key> != null)` before reading any field, and
`group_by(.epoch) | map(.[-1])` to get one value per epoch.

## Curated metrics — pull what's meaningful, not everything

The project logs ~50 keys; most are noise for a run diagnosis. Start from this
curated set and **refine it after each run** — when a metric turns out
non-discriminating (same across healthy/broken runs) drop it; when a question
needed a key not listed, add it. Keep this list current; it's the whole point of
not pulling everything.

**Meaningful (pull these):**
- `val_mse_loss`, `mse_loss` — the fit and the train/val gap (the headline).
- `latent/std_mean` (σ), `latent/mu_std` (spacing) — the cloud-overlap question.
- `latent/kld_per_dim_median`, `latent/kld_alpha` — rate + where in the anneal.
- `latent/active_dims` — collapse check.
- `kld_loss` / `val_kld_loss` — total rate (corroborates kld/dim).
- `actmon/...encoder.../max_abs`, `actmon/...decoder.../max_abs`,
  `.../has_nan`, `.../has_inf` — instability vs memorisation discriminator.

**Usually skip (unless a specific question needs them):**
- Image panels (`Diff`, `Ground Truth`, `Predictions`, `PCA`, `MSE Per Transform`)
  — heavy, not numeric; only fetch if doing a visual check.
- `permuter/*`, `perm_diag/*`, `permutation_loss` — **permuter is dormant**; flat
  zeros. Pull only once the permuter is active.
- `signal_to_noise_ratio_loss`, `mae_loss`, `beta_recon`, `gamma_recon`,
  `alpha_recon`, `tau` — loss-term weights/aux; rarely move the diagnosis.
- LR keys (`CosineWarmupLR/*`, `OneCycleLR`) — only when chasing an instability.

**Not logged (don't waste a call):** `grad_norm` is **not** logged in this project
— a `keys=["grad_norm"]` fetch returns empty `rows`. To classify a suspected
instability without it, lean on `mse_loss` (train) vs `val_mse_loss` (a train-flat
/ val-diverging split is *generalisation* failure, not a train-time blow-up) and
the offline activation `max_abs` / `report.json` reconstruction R². If you need
`grad_norm`, it has to be added to the training logging first.

**Observed log cadence (PIGVAE, 40-epoch run ≈ 1700 steps):** `val_*` once/epoch
(~43 steps apart); `latent/*` a few times/epoch; dense per-step losses + actmon
every step. ~42–43 steps ≈ 1 epoch is a good step→epoch conversion.
