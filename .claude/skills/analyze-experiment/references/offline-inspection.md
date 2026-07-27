# Offline Checkpoint Inspection (szary / SLURM)

`scripts/diagnose_model.py --inspect` loads a checkpoint and reports model
internals that wandb doesn't log: weight/activation stats, attention (QK-norm
gains, entropy), latent eff-rank, and an image-space reconstruction decomposition
(PCA-truncation floor vs model error, R², per-PC error). Use it to *confirm a
mechanism* the curves only hint at — e.g. memorisation (residual `max_abs`
blow-up, R² drop) or latent collapse (eff-rank).

It is JSON-first: it prints a JSON blob to stdout and writes artifacts to an
out-dir. Capture stdout to a file and read the JSON.

## Critical: where it runs

- **Compute node only.** The script needs the GPU/data environment and reads
  checkpoints under `/raid_encrypted/immucan/embeddings/tnocon/...`, which is
  reliably accessible **from inside a szary allocation**, not from the login node.
- Run it via `srun` (synchronous, one shot) — *not* an interactive `--pty bash`,
  which can't be driven from a single command.
- Use `uv run python ...` (the project's env manager), from the repo root.

## Step A — resolve the run → checkpoint dir

A wandb run (e.g. `jtscgh92`) maps to a hydra run folder named by **timestamp**:
`/raid_encrypted/immucan/embeddings/tnocon/logs/train/runs/<YYYY-MM-DD_HH-MM-SS>/checkpoints/`.

The match is **by date and is fuzzy** — the folder time is close to, usually at or
just after, the wandb `createdAt`, but not identical. Resolve it like this:

1. Get the run's `createdAt` (and name) from the wandb MCP.
2. List the runs dir and pick the folder closest to that time. Do the listing
   **inside the allocation** (authoritative); only try it on the login node first
   as an optimisation, and never assume login-node access succeeds:
   ```bash
   srun --qos=tnocon --partition=common --cpus-per-task=2 --mem=4G --time=0:10:00 \
     bash -lc "ls -dt /raid_encrypted/immucan/embeddings/tnocon/logs/train/runs/*/ | head -20"
   ```
3. If two folders are plausibly the run (close timestamps), **show the candidates
   and confirm with the user** before inspecting — a wrong checkpoint silently
   produces a wrong diagnosis.
4. In `<run>/checkpoints/` identify both:
   - `last.ckpt` — the end-state (overfit/collapsed, if it went bad).
   - the **best-on-val** checkpoint (e.g. `epoch_*.ckpt` / `best*.ckpt`) — the
     healthy state. `ls` the dir to see the actual filenames.

## Step B — the `--experiment` config

Pass the experiment config whose `model` overrides match the run's architecture
(`--experiment vae16_lowkl`). Infer it from the run's wandb name/group/tags/config
(e.g. a run tagged/named `vae16-lowkl` → `configs/experiment/vae16_lowkl.yaml`).
If unsure which config, confirm with the user — architecture mismatch makes the
load fail or, worse, load wrong.

## Step C — run the inspector (best + last)

Run in the **background** (queue wait + inspection can exceed a foreground
timeout), then read the JSON when it finishes. One invocation per checkpoint:

```bash
REPO=/home/tnocon/master_thesis/IMMUVIS-PIGVAE
OUT=$REPO/logs/diagnostics/<run-id>-<best|last>
mkdir -p "$OUT"
srun --qos=tnocon --partition=common --cpus-per-task=6 --mem=20G --time=1:00:00 \
  bash -lc "cd $REPO && uv run python scripts/diagnose_model.py \
    --ckpt /raid_encrypted/immucan/embeddings/tnocon/logs/train/runs/<TS>/checkpoints/<CKPT> \
    --paths szary --split val --inspect --experiment <EXP> --out-dir $OUT" \
  > "$OUT/diagnose.json" 2>&1
```

The out-dir holds **two** JSON files plus a `plots/` dir — know which to read:
- `$OUT/diagnose.json` — the captured stdout: human-readable per-batch report
  (latent, activation stats, decoder-std diagnostic, equivariance, node
  inspection). Good for a quick scan; the stream mixes progress logs with the
  report, so read from the tail.
- `$OUT/report.json` — the **structured** blob (top-level keys `meta`, `weights`,
  `activations`, `attention`, `latent`, `reconstruction`, `flags`). **This is where
  the reconstruction decomposition lives** — `diagnose.json` does *not* contain it.
  Parse this for any recon/R² number (see Step D).
- `$OUT/plots/` — `per_block_max_abs.png`, `per_channel_r2.png`.

Useful flags: `--split val|test`, `--num-batches N` (default 3), `--batch-idx K`
(a specific batch), `--tau` (temperature; defaults from checkpoint), `--data-dir`
to override. Defaults are fine for a standard diagnosis.

## Step D — what to pull from the JSON

Tie each number back to the curve-level question:
- **Residual `max_abs`** (pre-norm, enc/dec): blow-up vs the AE baseline (~18)
  corroborates memorisation/instability at `last` vs `best`.
- **QK-norm gains**: drifting gains = attention sharpening; cross-check the online
  activation monitor.
- **Latent eff-rank** (use 90%-energy rank, the honest measure — *not*
  participation ratio, which exaggerates low-rank): collapse vs healthy use.
- **Reconstruction (in `report.json` → `reconstruction`, *not* `diagnose.json`)**:
  - `overall_mse` — the PCA-coeff reconstruction error; it **matches the run's
    wandb `val_mse`**, so it's the offline way to *confirm a checkpoint actually
    reconstructs at the curve's value* (e.g. confirmed fb0.2 `last` really is
    0.34, not a logging artifact). Use it to verify a suspicious wandb number.
  - `image_space.image_r2_mean` / `image_r2_min`, and the `vs_input` block:
    `pca_floor_mse` (the data/front-end floor, *identical across runs on the same
    data*) and `model_added_mse` (everything the decoder adds on top). The split
    `pca_floor_mse` + `model_added_mse` is the image-space decomposition.
  - **Smoking-gun read for decoder overfit/brittleness**: identical `pca_floor_mse`
    but a much larger `model_added_mse`, and a **negative `image_r2_min`** (worse
    than predicting the mean on some images) at `last` — that's a brittle, memorised
    decoder, *not* over-regularisation or collapse (those keep R² ≥ 0 and move the
    latent stats). Cross-check against a healthy sibling's `last`.
- **Latent vs decoder dissociation**: if `last`'s latent stats (σ, `mu_std`,
  kld/dim, `active_dims`) look ~identical to a healthy sibling but `overall_mse` is
  far worse, the failure is in the **decoder mapping**, not the latent — aggregate
  latent health can't see it. Encoder `inter_node_cos_sim_mean` is a useful
  side-channel (lower = more position-distinct codes = more for the decoder to
  memorise).

The **best-vs-last contrast** (and the **broken-vs-healthy-sibling `last`**
contrast) is the deliverable: healthy checkpoint numbers next to end-state numbers
make the mechanism concrete.

## sbatch fallback (congested `common` partition)

If `srun` queues too long, submit a batch job instead and poll for the output
file. Write a small job script to the scratchpad, `sbatch` it, then check for
completion (poll on the output file existing / job state). Same command body as
Step C inside the script:

```bash
#!/bin/bash
#SBATCH --qos=tnocon
#SBATCH --partition=common
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --output=<OUT>/diagnose.%j.out
cd /home/tnocon/master_thesis/IMMUVIS-PIGVAE
uv run python scripts/diagnose_model.py --ckpt <CKPT> --paths szary \
  --split val --inspect --experiment <EXP> --out-dir <OUT>
```

Prefer `srun` for the common single-checkpoint case (one step, synchronous);
reach for `sbatch` when the queue is slow or when diagnosing several checkpoints.
