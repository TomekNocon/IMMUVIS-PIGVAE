"""Thesis figure (Sec. 3.5.1): four marker channels of one 128x128 cords crop, side by side.

Preprocessing matches the encoder input: arcsinh(x/5) -> Butterworth low-pass (0.2) -> clip(0,4)/4.
For display, each channel is additionally contrast-stretched to its 1st-99.5th percentile (stated
in the caption). Crops are the non-overlapping 128 tiling the embeddings were cut with.

Stage 1 (--scan): score crops over many images, write a candidate contact sheet.
Stage 2 (--image ID --crop K): render the final figure as PDF + PNG.
"""
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import argparse, glob, os
import numpy as np, tifffile, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.filters import butterworth

ROOT = "/raid_encrypted/immucan/immuvis_split/train/cords/imgs"
PANEL = "/home/tnocon/multiplex-image-model/configs/all_panels_config.yaml"
# (marker, display colour): nuclei, tumour epithelium, immune cells, stroma
SHOW = [("DNA1", "#4C9BE8"), ("panCK", "#E8C34C"), ("CD45", "#E8584C"), ("SMA", "#5CCB7A")]
COLOURS = {"DNA1": "#4C9BE8", "panCK": "#E8C34C", "CD45": "#E8584C", "SMA": "#5CCB7A",
           "CD8a": "#E04CC8", "CD4": "#4CD8E0", "CD3": "#F08A3C", "CD68": "#B07CF0"}
P = 128

def markers():
    return yaml.safe_load(open(PANEL))["markers"]["cords"]

def preprocess(img):
    img = np.arcsinh(img.astype(np.float32) / 5.0)
    img = np.stack([butterworth(c, cutoff_frequency_ratio=0.2, high_pass=False) for c in img])
    return np.clip(img, 0, 4) / 4

def crops(img):
    _, H, W = img.shape
    out = []
    for i in range(0, H - P + 1, P):
        for j in range(0, W - P + 1, P):
            out.append(((i, j), img[:, i:i + P, j:j + P]))
    return out

def stretch(ch, lo=1, hi=99.5):
    a, b = np.percentile(ch, lo), np.percentile(ch, hi)
    return np.clip((ch - a) / (b - a), 0, 1) if b > a else np.zeros_like(ch)

def tint(ch, hexcol):
    rgb = np.array(matplotlib.colors.to_rgb(hexcol))
    return stretch(ch)[..., None] * rgb

def score(crop, idx):
    # want every shown channel to have real signal, covering a decent fraction of the crop
    s = []
    for k in idx:
        c = crop[k]
        frac = (c > np.percentile(c, 50) + 0.05).mean()
        s.append(min(frac, 0.5) * c.std())
    return min(s)  # the weakest channel decides

def scan(n_images, out):
    names = markers(); idx = [names.index(m) for m, _ in SHOW]
    files = sorted(glob.glob(f"{ROOT}/*.tiff"))
    rng = np.random.default_rng(0)
    files = list(rng.choice(files, size=min(n_images, len(files)), replace=False))
    cands = []
    for f in files:
        img = preprocess(tifffile.imread(f))
        for k, (ij, c) in enumerate(crops(img)):
            cands.append((score(c, idx), os.path.basename(f)[:-5], k, ij, c[idx]))
    cands.sort(key=lambda t: -t[0])
    top = cands[:12]
    n = len(SHOW)
    fig, axes = plt.subplots(len(top), n, figsize=(2 * n, 2 * len(top)))
    for r, (s, iid, k, ij, cs) in enumerate(top):
        for c, (m, col) in enumerate(SHOW):
            axes[r, c].imshow(tint(cs[c], col)); axes[r, c].axis("off")
            if r == 0: axes[r, c].set_title(m)
        axes[r, 0].text(-5, 64, f"{iid}\ncrop {k}", ha="right", va="center", fontsize=8)
    tag = "_".join(m for m, _ in SHOW)
    plt.tight_layout(); plt.savefig(f"{out}/candidates_{tag}.png", dpi=80)
    for s, iid, k, ij, _ in top: print(f"score={s:.4f} image={iid} crop={k} at={ij}")

def render(image_id, k, out):
    names = markers(); idx = [names.index(m) for m, _ in SHOW]
    img = preprocess(tifffile.imread(f"{ROOT}/{image_id}.tiff"))
    (i, j), c = crops(img)[k]
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    n = len(SHOW)
    fig, axes = plt.subplots(1, n, figsize=(6.3, 6.3 / n * 1.18))  # ~ A4 text width
    for ax, (m, col), ci in zip(axes, SHOW, idx):
        ax.imshow(tint(c[ci], col), interpolation="nearest")
        ax.set_title(m, fontsize=9, pad=3); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
    # IMC is acquired at 1 um/px, so 50 px = 50 um
    axes[0].plot([P - 8 - 50, P - 8], [P - 8, P - 8], color="white", lw=2, solid_capstyle="butt")
    axes[0].text(P - 8 - 25, P - 12, "50 µm", color="white", ha="center", va="bottom", fontsize=7)
    plt.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.01, wspace=0.04)
    base = f"{out}/fig_cords_crop_{image_id}_c{k}_" + "_".join(m for m, _ in SHOW)
    plt.savefig(base + ".pdf"); plt.savefig(base + ".png", dpi=300)
    print("wrote", base, "origin", (i, j), "channels in tiff:", img.shape[0])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=0, help="number of images to scan")
    ap.add_argument("--image"); ap.add_argument("--crop", type=int)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--markers", default="DNA1,panCK,CD45,SMA", help="comma-separated, see COLOURS")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    SHOW = [(m, COLOURS[m]) for m in a.markers.split(",")]
    scan(a.scan, a.out) if a.scan else render(a.image, a.crop, a.out)
