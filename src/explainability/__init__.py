"""Post-hoc explainability for the PIGVAE encoder and the downstream gated-ABMIL.

Two levels (see the module docstrings):
- `encoder_attribution` — Integrated Gradients over the encoder input, i.e. *what the
  learned representation (z_global) encodes*, optionally mapped from PCA space back to
  the original marker channels.
- `abmil_attribution` — gated-ABMIL attention weights + IG over the crop bag, i.e. *which
  crops drive the clinical decision* (added alongside).

Method choice and comparison: Captum Integrated Gradients is the robust workhorse (no
per-layer LRP rules needed for the custom graph transformer); `rollout` is a cheap
attention baseline; `faithfulness` validates any heatmap via deletion/insertion.
"""
