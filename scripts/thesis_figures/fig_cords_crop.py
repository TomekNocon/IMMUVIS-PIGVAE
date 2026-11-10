"""Marker channels of one 128x128 cords crop, side by side (Sec. 3.5.1).

Preprocessing matches the encoder input (arcsinh(x/5) -> Butterworth low-pass (0.2) -> clip(0,4)/4);
for display each channel is contrast-stretched to its 1st-99.5th percentile (stated in the caption).
Crops are the non-overlapping 128-px tiling the embeddings were cut with. Runs on szary (raid TIFFs).

Usage:
    # 1) score crops over many images and write a contact sheet of candidates
    uv run python scripts/thesis_figures/fig_cords_crop.py scan --n-images 40
    # 2) render the chosen crop
    uv run python scripts/thesis_figures/fig_cords_crop.py render --image 7775 --crop 17 --markers DNA1,panCK,CD8a,SMA
"""

import argparse
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import matplotlib.pyplot as plt
import numpy as np

from figlib import imc, paths, style

P = imc.CROP
DEFAULT_MARKERS = "DNA1,panCK,CD45,SMA"  # nuclei, tumour epithelium, immune cells, stroma


def crop_score(crop: np.ndarray, channels: list[int]) -> float:
    """How well the weakest shown channel is covered by real signal (higher is better)."""
    scores = []
    for k in channels:
        c = crop[k]
        covered = (c > np.percentile(c, 50) + 0.05).mean()
        scores.append(min(covered, 0.5) * c.std())
    return min(scores)


def scan(markers: list[str], n_images: int, out_dir) -> None:
    """Score every crop of ``n_images`` random slides and write the top-12 as a contact sheet."""
    channels = [imc.cords_markers().index(m) for m in markers]
    files = sorted(paths.CORDS_TRAIN_TIFFS.glob("*.tiff"))
    rng = np.random.default_rng(0)
    files = list(rng.choice(files, size=min(n_images, len(files)), replace=False))
    candidates = []
    for f in files:
        for k, (ij, crop) in enumerate(imc.crops(imc.load_slide(f))):
            candidates.append((crop_score(crop, channels), f.stem, k, ij, crop[channels]))
    candidates.sort(key=lambda t: -t[0])
    top = candidates[:12]

    fig, axes = plt.subplots(len(top), len(markers), figsize=(2 * len(markers), 2 * len(top)))
    for r, (_, image, k, _, crop) in enumerate(top):
        for c, m in enumerate(markers):
            axes[r, c].imshow(imc.tint(crop[c], style.MARKER_COLOURS[m]))
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(m)
        axes[r, 0].text(-5, 64, f"{image}\ncrop {k}", ha="right", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"candidates_{'_'.join(markers)}.png", dpi=80)
    for score, image, k, ij, _ in top:
        print(f"score={score:.4f} image={image} crop={k} at={ij}")


def render(markers: list[str], image: str, k: int, out_dir) -> None:
    """The final figure: one panel per marker with a 50 um scale bar."""
    channels = [imc.cords_markers().index(m) for m in markers]
    slide = imc.load_slide(paths.CORDS_TRAIN_TIFFS / f"{image}.tiff")
    (i, j), crop = imc.crops(slide)[k]
    style.use_serif(9, mathtext=False)
    n = len(markers)
    fig, axes = plt.subplots(1, n, figsize=(6.3, 6.3 / n * 1.18))
    for ax, m, ch in zip(axes, markers, channels, strict=True):
        ax.imshow(imc.tint(crop[ch], style.MARKER_COLOURS[m]), interpolation="nearest")
        ax.set_title(m, fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    # IMC is acquired at 1 um/px, so 50 px = 50 um
    axes[0].plot([P - 8 - 50, P - 8], [P - 8, P - 8], color="white", lw=2, solid_capstyle="butt")
    axes[0].text(P - 8 - 25, P - 12, "50 µm", color="white", ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.01, wspace=0.04)
    stem = out_dir / f"fig_cords_crop_{image}_c{k}_{'_'.join(markers)}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    print(f"wrote {stem} | crop origin {(i, j)} | {slide.shape[0]} channels in the TIFF")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["scan", "render"])
    ap.add_argument(
        "--markers",
        default=DEFAULT_MARKERS,
        help=f"comma-separated, from {list(style.MARKER_COLOURS)}",
    )
    ap.add_argument(
        "--n-images", type=int, default=40, help="scan: number of random slides to score"
    )
    ap.add_argument("--image", help="render: image id, e.g. 7775")
    ap.add_argument("--crop", type=int, help="render: crop index in the 128-px tiling")
    ap.add_argument("--out-dir", default=str(paths.FIGURES))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = a.markers.split(",")
    if a.mode == "scan":
        scan(selected, a.n_images, out)
    else:
        render(selected, a.image, a.crop, out)
