# PIGVAE Diagnostics Playbook

The metric vocabulary, healthy ranges, and failure-mode fingerprints for the
per-node IMC VAE. Ranges are anchored to the runs already in the wiki — treat
them as reference points, not hard thresholds, and always re-anchor against the
sibling runs the wiki links for *this* recipe/resolution.

## Metric vocabulary (wandb keys → meaning)

| metric | meaning | read it for |
|---|---|---|
| `train mse` / `val_mse` | reconstruction error (PCA-coeff space) | fit + the train/val gap |
| `kld` / `kld_per_dim` (median) | KL of the per-node posterior to the prior | rate; how hard the prior pushes |
| `std_mean` (σ) | mean posterior std | how stochastic the code is; cloud width |
| `mu_std` | std of posterior means across nodes/images | **code spacing** between samples |
| `active_dims` | latent dims with non-trivial KL | collapse vs full use (of total = dims/node × nodes) |
| `kld_alpha` (α) | KL weight on the anneal schedule | whether you're in warmup (α<1) or stationary (α=1) |
| `lr`, `grad_norm` | optimiser state | instability fingerprints |
| activation-monitor maxima | online pre-norm residual `max_abs`, QK gains | blow-up during training |

Total latent dims = per-node `node_z_dim` × #nodes. 6×6 = 36 nodes; full
16×16 = 256 nodes. So z32@16×16 = 8192 dims, z16@16×16 = 4096, z32@6×6 = 1152.

## Reference floors (don't chase below these)

- **AE reconstruction floor**: ~0.022 at 16×16, ~0.040-ish regime at 6×6. A VAE
  near this on val is reconstructing as well as the deterministic AE.
- **PCA-128 truncation floor** (image-space): the 768→128 PCA front-end bakes a
  floor into the *targets* — ~62% of total image-space error in the decomposition.
  The lever for image fidelity is *more PCA components*, not the encoder/decoder.
- **Data noise floor**: the whitened PCA tail; residual concentrates at **edges**
  and **high-magnitude** nodes. Context levers (radius/ALiBi) proved weak against it.

Implication: a *reconstruction* plateau is often floor-bound, not a model failure.
Don't prescribe recon tuning when the gap to the floor is already small — pivot to
the generalisation/latent question.

## Reading fit: overfit vs underfit vs healthy

Look at the **whole** curve, especially post-warmup (α=1):

- **Memorisation / overfit**: val_mse climbs in the back half while **train stays
  flat**, gap opens monotonically. Diagnostic tell: the collapse happens at a
  **stationary latent** (α=1, `kld/dim`, σ, `mu_std` all flat) — so it's pure
  encoder/decoder memorisation, *not* rate-distortion. (jtscgh92: 0.026→0.071,
  train flat ~0.033.)
- **Underfit**: best val_mse worse than a sibling, train can't go low either
  (train flat *high*), small gap. Over-correction — e.g. latent too small or KL
  too strong. (42r3jeg4: best only 0.042, train ~0.052, gap 0.013.)
- **Healthy**: val near the AE floor with a small, stable gap and a fully-used,
  bounded latent. (vae6_lowkl: val 0.040, active 1152/1152.)

Bias–variance bracket: a memorising run and an underfitting run *bracket* the
curve; the endpoints can coincidentally tie. The win is the **best** checkpoint,
not the endpoint — confirm a best-on-val checkpoint exists.

## Classifying spikes (the key skill — don't just flag, diagnose)

- **KL-warmup overshoot (artifact, not damage)**: occurs *during the α ramp*
  (α rising). Signature: `mse↑` together with `kld↓`, recovering within a few
  epochs. It's the KL term suddenly biting as α grows, then the model adapts.
  (vae6_lowkl ep3: mse↑0.167 / kld↓0.28, recovered by ep5.) Report as transient.
- **Real instability**: `grad_norm` spike, NaN/inf, or activation-monitor
  `max_abs` blow-up; often LR- or clipping-related. This *is* damage. Check the
  grad-clipping band-aid in `configure_gradient_clipping`.
- **Fast-drop-then-plateau (not a spike, not LR)**: the early steep MSE drop to a
  plateau is the **PCA-128 floor + leading-PC dominance** (zscore=False), not a
  learning-rate effect. A stable run with no instability fingerprint that drops
  fast and flattens is hitting the floor, behaving correctly.

When you see a spike, state which of the three it is and the evidence (what α,
kld, grad_norm were doing at that step).

## Latent health: the cloud-overlap picture

The central question for *generalisation* of this latent:

- **σ vs code spacing**: compare `std_mean` (σ, cloud width) to `mu_std` (spacing
  between codes). If **σ ≪ spacing**, the training Gaussians don't overlap, so the
  decoder can map each isolated code to its exact target → **memorisation**. If σ
  is comparable to spacing, codes overlap and the decoder must generalise.
  (jtscgh92 memorised: σ 0.41 vs spacing 0.82.)
- **Free-bits as the looseness**: `free_bits` × total dims = free nats the prior
  doesn't constrain. A latent carrying thousands of nats (≫ the ~11 nats needed to
  index 40k images) is acting as a **pointer into decoder weights** — so shrinking
  `node_z_dim` can't fix memorisation (confirmed). The lever is **`free_bits`
  down** (e.g. 0.5→0.1→0), tightening the prior so σ grows toward the spacing.
- **AE↔VAE spectrum**: low KL pressure → near-AE (great representation, loose
  prior match, weak sampling); high pressure → over-regularised (val_mse rises,
  dims squeezed uniformly, *not* collapsed) or, at the extreme, posterior
  collapse (`active_dims` drops, kld/dim→0). vae6 (scale 0.05, fb 0) sat at
  over-regularised (val 0.167); vae6_lowkl (scale 0.01, fb 0.5) sat near the AE
  end and healthy.

Over-regularised ≠ collapsed: over-regularised keeps dims *active* but squeezed
(kld/dim small *and uniform*, val_mse up); collapse *drops* active_dims.

## Choosing the next lever

Name one lever and justify why it beats the alternatives:
- Memorisation at a stationary latent → **lower `free_bits`** (tighten prior so
  clouds overlap), *not* fewer dims (that only trades overfit for underfit).
- Over-regularised → **lower KL `scale`** and/or add `free_bits` to protect dims.
- Recon plateau near the floor → stop tuning recon; pivot to the latent /
  generalisation question, or (for image fidelity only) more PCA components.
- Always close by tying the lever to the relevant `synthesis.md` open question —
  especially "does a more-regularised z transfer better, or just reconstruct
  worse?", which only downstream transfer (not recon) can settle.
