"""Hydra entrypoint for professional IMC raw-input visualisation.

Renders a whole-image ``.tiff`` or a pre-cut ``.npy`` patch two publication-standard ways --
a per-marker contact sheet and a false-colour composite -- and saves PNGs to ``out_dir``. Reads
raw pixels only (no model/checkpoint), so it runs CPU-only on szary alongside anything else.

Examples:
  uv run scripts/visualize_imc.py input_path=/raid_.../imgs/foo.tiff paths=szary
  uv run scripts/visualize_imc.py input_path=/raid_.../foo.tiff figure=composite \
      panel_config=/raid_.../all_panels_config2.yaml dataset=cords \
      composite_channels=[DNA1,CD8,panCK] paths=szary
"""

import os

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
from omegaconf import DictConfig, OmegaConf

from src.explainability import imc_viz


def _marker_names(cfg: DictConfig):
    """Resolve marker names from an explicit list, else a panel-config yaml, else None."""
    if cfg.get("marker_names"):
        return list(cfg.marker_names)
    if cfg.get("panel_config") and cfg.get("dataset"):
        panels = OmegaConf.load(cfg.panel_config)
        names = panels["markers"][cfg.dataset]
        return list(OmegaConf.to_container(names, resolve=True))
    return None


@hydra.main(version_base="1.3", config_path="../configs", config_name="visualize_imc")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    img = imc_viz.load_imc_image(cfg.input_path, arcsinh_scale=cfg.arcsinh_scale)
    names = _marker_names(cfg)
    C, H, W = img.shape
    print(f"[imc-viz] {cfg.input_path} -> [C={C}, H={H}, W={W}]"
          f"{' (named)' if names else ' (unnamed markers)'}")

    stem = cfg.name or os.path.splitext(os.path.basename(cfg.input_path))[0]
    if cfg.get("patch") is not None:
        patches, coords = imc_viz.get_all_patches(img, int(cfg.patch_size))
        pi = int(cfg.patch)
        img = patches[pi]
        stem = f"{stem}_patch{pi}"
        print(f"[imc-viz] patch {pi}/{len(patches)} at {coords[pi]}")

    pctl = tuple(cfg.pctl)
    if cfg.figure in ("both", "grid"):
        out = os.path.join(cfg.out_dir, f"{stem}_grid.png")
        imc_viz.plot_marker_grid(img, marker_names=names, out_path=out, cmap=cfg.cmap,
                                 pctl=pctl, ncols=int(cfg.ncols),
                                 pixel_size_um=cfg.pixel_size_um, scale_bar_um=cfg.scale_bar_um,
                                 title=f"{stem} · {C} markers")
        print(f"[imc-viz] wrote {out}")

    if cfg.figure in ("both", "composite"):
        chans = (list(cfg.composite_channels) if cfg.get("composite_channels")
                 else imc_viz.default_composite_channels(names, num_channels=C, k=int(cfg.composite_k)))
        out = os.path.join(cfg.out_dir, f"{stem}_composite.png")
        imc_viz.plot_false_colour(img, chans, marker_names=names, out_path=out, pctl=pctl,
                                  blend=cfg.blend, pixel_size_um=cfg.pixel_size_um,
                                  scale_bar_um=cfg.scale_bar_um, title=f"{stem} · composite")
        print(f"[imc-viz] wrote {out} (channels={chans})")

    print(f"[imc-viz] done -> {os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    main()
