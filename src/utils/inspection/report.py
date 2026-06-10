# src/utils/inspection/report.py
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.switch_backend("Agg")

# Flag thresholds (tune here).
MAX_ABS_FLAG = 50.0
GAIN_RMS_LO, GAIN_RMS_HI = 0.5, 2.0
RANK_RATIO_FLAG = 0.5
R2_FLAG = 0.3


def build_flags(results: dict) -> list[str]:
    flags: list[str] = []
    for name, w in results.get("weights", {}).items():
        if "gain_rms" in w and not (GAIN_RMS_LO <= w["gain_rms"] <= GAIN_RMS_HI):
            flags.append(f"[weights] {name}: gain_rms={w['gain_rms']:.3f} outside [{GAIN_RMS_LO},{GAIN_RMS_HI}]")
        if "rank_ratio" in w and w["rank_ratio"] < RANK_RATIO_FLAG:
            flags.append(f"[weights] {name}: low rank_ratio={w['rank_ratio']:.2f}")
    for name, a in results.get("activations", {}).items():
        if a.get("max_abs", 0) > MAX_ABS_FLAG:
            # Per-block hooks capture the PRE-final-norm residual stream, which is large by
            # design (output_init_std) and bounded at the module output by final_norm — check
            # the `...graph_transformer` (post-norm) entry before treating this as instability.
            note = " (pre-norm residual; see post-norm graph_transformer entry)" if ".blocks." in name else ""
            flags.append(f"[activations] {name}: max_abs={a['max_abs']:.1f} > {MAX_ABS_FLAG}{note}")
    lat = results.get("latent", {})
    if lat:
        # Flag on the honest energy-rank (90% of latent variance), not the participation
        # ratio (which over-concentrates and under-reports usable rank).
        if lat.get("energy_rank90_ratio", 1.0) < RANK_RATIO_FLAG:
            flags.append(f"[latent] energy_rank90={lat.get('energy_rank90')}/{lat.get('z_dim')} "
                         f"holds 90% of variance (bottleneck under-used)")
        if lat.get("active_dims", lat.get("z_dim", 0)) < 0.5 * lat.get("z_dim", 1):
            flags.append(f"[latent] only {lat['active_dims']}/{lat['z_dim']} active dims")
    rec = results.get("reconstruction", {})
    bad = [i for i, r2 in enumerate(rec.get("per_channel_r2", [])) if r2 < R2_FLAG]
    if bad:
        flags.append(f"[reconstruction] {len(bad)} channels with R2<{R2_FLAG}: {bad[:15]}")
    return flags


def _save_plots(results: dict, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    rec = results.get("reconstruction", {})
    if rec.get("per_channel_r2"):
        fig, ax = plt.subplots()
        ax.plot(rec["per_channel_r2"])
        ax.set(title="Per-channel R²", xlabel="channel", ylabel="R²")
        fig.savefig(plot_dir / "per_channel_r2.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
    acts = results.get("activations", {})
    maxes = {k: v.get("max_abs", 0) for k, v in acts.items() if "block" in k.lower()}
    if maxes:
        fig, ax = plt.subplots()
        ax.bar(range(len(maxes)), list(maxes.values()))
        ax.set_xticks(range(len(maxes)))
        ax.set_xticklabels(list(maxes.keys()), rotation=90, fontsize=6)
        ax.set(title="Per-block max_abs")
        fig.savefig(plot_dir / "per_block_max_abs.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def _render_md(results: dict, flags: list[str]) -> str:
    lines = ["# Model inspection report", ""]
    meta = results.get("meta", {})
    if meta:
        lines += ["## Meta", "```json", json.dumps(meta, indent=2), "```", ""]
    lines += ["## Flags", ""]
    lines += [f"- {f}" for f in flags] if flags else ["- (none)"]
    lat = results.get("latent", {})
    if lat:
        lines += ["", "## Latent",
                  f"- active_dims: {lat.get('active_dims')}/{lat.get('z_dim')}",
                  f"- **energy rank** (honest): {lat.get('energy_rank90')}/{lat.get('z_dim')} hold 90%, "
                  f"{lat.get('energy_rank99')}/{lat.get('z_dim')} hold 99% of latent variance",
                  f"- participation ratio (over-concentrates): {lat.get('effective_rank'):.1f}"
                  f"/{lat.get('z_dim')}"]
    film = results.get("film", {})
    if film:
        lines += ["", "## FiLM conditioning (decoder)",
                  f"- bound (tanh): {film.get('bound')}",
                  f"- |gamma| max: {film.get('gamma_absmax_overall'):.3f}  "
                  f"|beta| max: {film.get('beta_absmax_overall'):.3f}",
                  "  (bound=True should keep both <= 1.0; >1 means the tanh bound did NOT fire)"]
    rec = results.get("reconstruction", {})
    if rec:
        lines += ["", "## Reconstruction (PCA-coefficient space)",
                  f"- overall_mse: {rec.get('overall_mse')}",
                  f"- worst channels: {rec.get('worst_channels')}"]
        im = rec.get("image_space")
        if im:
            lines += ["", "## Reconstruction (image space, inverse-PCA)",
                      f"- n_orig_channels: {im['n_orig_channels']}",
                      f"- vs inverse(target) [model error]: mse={im['image_mse']:.5f} "
                      f"mae={im['image_mae']:.5f} R²_mean={im['image_r2_mean']:.3f}"]
            vi = im.get("vs_input")
            if vi:
                lines += [f"- **vs INPUT x [full pipeline]**: mse={vi['image_mse_vs_input']:.5f} "
                          f"mae={vi['image_mae_vs_input']:.5f} "
                          f"R²_global={vi.get('image_r2_vs_input_global', float('nan')):.3f} "
                          f"R²_mean={vi['image_r2_vs_input_mean']:.3f}",
                          f"- PCA-128(+clip) floor (inverse(target) vs x): mse={vi['pca_floor_mse']:.5f} "
                          f"R²_global={vi.get('pca_floor_r2_global', float('nan')):.3f} "
                          f"R²_mean={vi['pca_floor_r2_mean']:.3f}",
                          "  (R²_global = variance-weighted ≈ PCA explained-variance; "
                          "R²_mean = per-channel unweighted, reads lower)",
                          f"- model adds over the floor: {vi['model_added_mse']:.5f}"]
    return "\n".join(lines) + "\n"


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def write_report(results: dict, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flags = build_flags(results)
    results = dict(results)
    results["flags"] = flags
    (out_dir / "report.json").write_text(json.dumps(_to_jsonable(results), indent=2))
    (out_dir / "report.md").write_text(_render_md(results, flags))
    _save_plots(results, out_dir / "plots")
