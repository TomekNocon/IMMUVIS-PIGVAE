---
name: analyze-experiment
description: >-
  Diagnose and analyze PIGVAE training runs from Weights & Biases (entity
  tomeknocon, project PIGVAE) for the IMC permutation-invariant VAE thesis. Use
  this whenever the user wants to understand what happened in a run — "analyze
  the last experiment", "what happened in this run", "did it overfit / underfit",
  "why did the loss spike", "why did val_mse climb", "is the latent healthy",
  "compare these runs / ablation", "what's the next step / how do I fix this" —
  even if they don't name wandb explicitly. It pulls run history + config from
  wandb, grounds the analysis in the research-ml wiki (prior findings + the live
  thesis), distinguishes real instability from KL-warmup artifacts, judges
  overfit/underfit and latent health, optionally runs the offline checkpoint
  inspector on szary for latent/attention internals, and proposes write-back
  entries for the research-ml wiki. Trigger it for any question about a PIGVAE
  experiment's losses, KL, latent metrics, generalisation, or next step.
---

# Analyze Experiment

Turn a PIGVAE training run (or a set of them) into a grounded diagnosis: what
worked, what failed, *why*, and the single most informative next lever — then
offer to record the finding in the research-ml wiki.

This model is a **permutation-invariant graph VAE for IMC images** with a
**per-node latent**. The permuter (the invariance objective) is currently
**dormant** — runs are a per-view AE/VAE substrate. The thesis question is
whether this substrate yields an *invariant, generalising* `z` that transfers
downstream, so **generalisation, not reconstruction, is the deciding metric.**
Keep that framing; it changes what "good" means.

## The two knowledge sources

- **wandb** — `tomeknocon/PIGVAE`. The ground truth for *this* run's curves,
  config, and logged latent metrics. Access via the wandb MCP tools.
- **research-ml wiki** — `/home/tnocon/master_thesis/research-ml`. The user's
  "brain": prior experiments, the evolving thesis (`synthesis.md`), and the
  diagnostic conventions. **Always read this first** so the analysis builds on
  what's already known instead of rediscovering it. New findings get written
  back here (see write-back step).

## Workflow

Work the steps in order. Create a todo per step so nothing is skipped.

### 1. Orient in the wiki (always first)

Read, in this order, before touching wandb:
- `research-ml/index.md` — the catalog; find the pages relevant to this run.
- `research-ml/synthesis.md` — the live thesis: open questions, current phase,
  what the next run was *supposed* to test.
- `research-ml/log.md` (tail) — recent ops: `grep "^## \[" log.md | tail -8`.
- The most relevant linked pages — especially `wiki/concepts/kl-regularization.md`
  and the latest `wiki/experiments/exp-*.md` / `wiki/sources/source-*.md`.

You're looking for: what recipe this run descends from, what hypothesis it tests,
the established healthy ranges, and which open question it speaks to.

### 2. Identify the run(s)

Default: the **most recent _finished_** run in `tomeknocon/PIGVAE`. List recent
runs via the wandb MCP (`query_wandb_tool`) and take the latest whose `state` is
`finished`, unless the user named one. **Never draw key takeaways from a `running`
run** — a mid-training run hasn't passed the α ramp or the back-half where
memorisation shows up, so its curves invite wrong conclusions. If the very latest
run is still `running`, say so in one line (it's useful context that the sweep is
in flight) but analyze the most recent finished run. Only analyze a running run if
the user explicitly asks for an in-progress check, and then label every takeaway
as provisional. Note the chosen run's `createdAt`, name, group, tags, and state.

**Auto-expand to an ablation** when the run only makes sense against a sibling —
e.g. a `free_bits` / `scale` / `node_z_dim` sweep, or a "does X fix the overfit
from run Y" follow-up. Pull the predecessor(s) the wiki points to and compare.
Don't over-pull: 1 run for a health check, 2–4 for a sweep.

### 3. Pull the data from wandb

For each run, get **config** (the levers) and **full history** (the curves):
- Config to surface: resolution / `node_z_dim` (→ total latent dims), KL `scale`,
  `free_bits`, the α schedule (`start`/`end` epochs), `zscore`, `clip`, epochs,
  batch size, LR.
- History (time series, not just summary): `train mse`, `val_mse`, `kld` and
  `kld_per_dim` (median), `std_mean` (σ), `mu_std` (code spacing), `active_dims`,
  `kld_alpha` (α), `lr`, and any activation-monitor maxima. (`grad_norm` is **not
  logged** in this project — see `references/wandb-history.md`; use the train-vs-val
  mse split + offline activation/R² to classify instability instead.)

Use `get_run_history_tool` for series and `compare_runs_tool` for ablations.
`diagnose_run_tool` can give a quick first pass. Pull the **whole** curve — the
back half (post-warmup) is where memorisation shows up.

**How to pull without getting a misleadingly coarse curve — read
`references/wandb-history.md` before fetching.** The short version: the history
tool caps each response (~75k chars) and downsamples, which washes out
once-per-epoch keys like `val_mse_loss` (you'll see ~4 of 40 points and miss the
overfit onset). Fetch **sparse per-epoch keys one at a time**
(`keys=["val_mse_loss"]` → all 40 epochs in one call); use full/windowed pulls for
dense per-step keys. Don't pull all ~50 logged keys — pull the **curated
meaningful set** in `wandb-history.md`, and refine that set after each run (drop
non-discriminating metrics, add any that a question needed).

### 4. Analyze — the diagnostic playbook

Read `references/diagnostics-playbook.md` and work through it. The core questions:

- **Reconstruction & generalisation.** Best `val_mse` and *when*; train vs val
  gap; does val climb in the back half while train stays flat (→ memorisation)?
  Place against the known floors (AE floor, PCA-truncation floor).
- **Spikes — classify, don't just flag.** Is a spike a **KL-warmup overshoot**
  (during the α ramp; `mse↑` with `kld↓`, recovers in a few epochs — *transient,
  not damage*) or **real instability** (`grad_norm` spike, NaN/inf, activation
  blow-up) or just the **fast-drop-to-floor** (PCA-128 + leading-PC dominance,
  not LR)? The playbook has the fingerprints.
- **Latent health.** σ (`std_mean`) vs code spacing (`mu_std`) — do the training
  clouds overlap (σ ≳ spacing) or not (→ decoder memorises)? `active_dims`,
  `kld_per_dim`, and where the run sits on the **AE↔VAE spectrum**
  (over-regularised vs collapse vs loose-bottleneck memorisation).
- **Connect to the thesis.** Which `synthesis.md` open question does this
  answer, and does it move the substrate→invariance→transfer story?

### 5. Offline checkpoint inspection (when internals are needed)

When the wandb metrics raise a question only the model internals can answer —
eff-rank, residual `max_abs`, QK-norm gains, image-space PCA-floor decomposition,
R², per-PC reconstruction — run the offline inspector on **szary**. This is a
SLURM compute node; the script will not run on the login node, and the
checkpoints live under `/raid_encrypted/...` which is read from the allocation.

Inspect **best-on-val and `last.ckpt`** — the contrast (healthy vs end-state) is
itself the diagnosis. Full procedure, run→checkpoint resolution, and the
sbatch fallback are in `references/offline-inspection.md`. Read it before running.

Skip this step for a quick health check or a pure-curves question.

### 6. Synthesize the report

Produce the report in chat using this structure:

```
# <run name / id> — <one-line verdict>

## Setup
Resolution, latent dims, KL (scale / free_bits / α schedule), epochs — and what
recipe it descends from + what it was meant to test.

## What happened
Recon: best val_mse @ epoch, train/val gap, overfit|underfit|healthy.
Spikes: each notable spike classified (warmup artifact | instability | floor).
Latent: σ vs spacing, active dims, kld/dim, AE↔VAE position.
[Offline inspection findings, if run.]

## Mechanism — why
The causal story, not just the symptom. (e.g. "σ 0.41 ≪ spacing 0.82 → clouds
don't overlap → decoder memorises in the back half".)

## What worked / what didn't
Bullet the wins and the failures, each tied to evidence.

## Next lever
The single most informative next step, and why it (not the alternatives) is the
lever. Tie to the relevant synthesis.md open question.
```

Lead with the verdict. Always give the *mechanism*, and always name the **one**
next lever rather than a menu — that's what lets the user act.

### 7. Propose write-back to the wiki

After the user has seen the report, **draft** the wiki updates and show them; only
write after approval (never auto-write). Match the existing format exactly —
read `references/wiki-writeback.md` for the templates and the per-file rules
(log.md entry, new `exp-*` page, `source-*` page, experiment-log table row,
`synthesis.md` update, index.md links). Convert relative dates to absolute.

## Notes

- Permuter dormant: do not interpret runs as having invariance pressure unless the
  config shows the permuter is active. If it just turned on, flag that explicitly.
- `zscore=True` has been tried and rejected (up-weights noise PCs) — don't
  recommend it as a fix without acknowledging that.
- The MEMORY note stands: never suggest a cross-entropy loss for the permuter.
- If wandb logs a config field that resolves the checkpoint dir, prefer it; the
  date-match in `offline-inspection.md` is the fallback.
