"""Extract the real data behind the pipeline figure for one cords crop (szary).

Saves the preprocessed slide, the crop origin, the stored ImmuVis 16x16x768 feature map of the crop
(train.h5) and its projection by the training PCA-128 (16x16 grid) to
``figures/_cache/pipeline_<image>_<row>_<col>.npz``. Consumed by fig_pipeline.py, fig_d4_views.py
and fig_architecture.py; extract_reconstruction.py adds the model's reconstruction.

Usage:
    uv run python scripts/thesis_figures/extract_pipeline_data.py --image 7775 --row 384 --col 256
"""

import argparse
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import h5py
import joblib
import numpy as np

from figlib import imc, paths


def stored_embedding(image: str, row: int, col: int) -> np.ndarray:
    """The crop's stored ImmuVis map [768, 16, 16]; tolerates either (row, col) order in positions."""
    with h5py.File(paths.CORDS_DATA / "train.h5", "r") as f:
        image_paths = np.array([p.decode() for p in f["paths"][:]])
        rows = np.where(np.char.endswith(image_paths, f"/{image}.tiff"))[0]
        positions = f["positions"][rows[0] : rows[-1] + 1]
        hit = [
            r for r, p in zip(rows, positions, strict=True) if int(p[0]) == row and int(p[1]) == col
        ]
        if not hit:
            hit = [
                r
                for r, p in zip(rows, positions, strict=True)
                if int(p[1]) == row and int(p[0]) == col
            ]
            print("matched with swapped (x, y) order")
        print("h5 row", hit[0], "position", f["positions"][hit[0]])
        return f["embeddings"][hit[0]]


def main(image: str, row: int, col: int, out_dir: str) -> None:
    slide = imc.load_slide(paths.CORDS_TRAIN_TIFFS / f"{image}.tiff")
    emb = stored_embedding(image, row, col)
    pca = joblib.load(paths.PCA_128)
    nodes = emb.reshape(768, -1).T  # 256 nodes x 768, row-major over the 16x16 grid
    z = (nodes - pca.mean_) @ pca.components_.T  # zscore=False, no clip, as in the final config
    print(
        f"PCA-128 explained variance {pca.explained_variance_ratio_.sum():.4f} | slide {slide.shape}"
    )
    out = Path(out_dir) / f"pipeline_{image}_{row}_{col}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        slide=slide.astype(np.float16),
        crop_ij=np.array([row, col]),
        emb=emb,
        z=z.T.reshape(128, 16, 16),
        evr=pca.explained_variance_ratio_,
        markers=np.array(imc.cords_markers()),
    )
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", default="7775")
    ap.add_argument("--row", type=int, default=384, help="crop origin row (pixels)")
    ap.add_argument("--col", type=int, default=256, help="crop origin column (pixels)")
    ap.add_argument("--out-dir", default=str(paths.CACHE))
    a = ap.parse_args()
    main(a.image, a.row, a.col, a.out_dir)
