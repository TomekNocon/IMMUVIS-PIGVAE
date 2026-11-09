"""Extract the real data behind the pipeline figure (Sec. 3.5.1) for one cords crop -> npz.

Runs on szary (raid). Pulls the whole slide, the 128x128 crop, the stored ImmuVis 16x16x768
feature map for that crop (train.h5), and its projection by the pipeline's PCA-128 (grid 16).
"""
import argparse, sys
import h5py, joblib, numpy as np, tifffile
sys.path.insert(0, "scripts/thesis_figures")
from fig_cords_crop import preprocess, markers, ROOT

D = "/raid_encrypted/immucan/embeddings/tnocon/data/IMC/cords"
ap = argparse.ArgumentParser()
ap.add_argument("--image", default="7775"); ap.add_argument("--i", type=int, default=384)
ap.add_argument("--j", type=int, default=256); ap.add_argument("--out", default="figures/_cache")
a = ap.parse_args()

slide = preprocess(tifffile.imread(f"{ROOT}/{a.image}.tiff"))
with h5py.File(f"{D}/train.h5", "r") as f:
    paths = np.array([p.decode() for p in f["paths"][:]])
    rows = np.where(np.char.endswith(paths, f"/{a.image}.tiff"))[0]
    pos = f["positions"][rows[0]:rows[-1] + 1]
    print("rows", rows[0], rows[-1], "positions sample:\n", pos[:6])
    hit = [r for r, p in zip(rows, pos) if int(p[0]) == a.i and int(p[1]) == a.j]
    if not hit:
        hit = [r for r, p in zip(rows, pos) if int(p[1]) == a.i and int(p[0]) == a.j]
        print("matched with swapped (x, y) order")
    r = hit[0]; emb = f["embeddings"][r]; print("row", r, "pos", f["positions"][r], "emb", emb.shape)

pca = joblib.load(f"{D}/pca_model_128_center_crop_16.pkl")
X = emb.reshape(768, -1).T                     # 256 nodes x 768 (row-major over the 16x16 grid)
Z = (X - pca.mean_) @ pca.components_.T        # zscore=False, clip=None in the final config
print("pca expl. var (128):", pca.explained_variance_ratio_.sum())
import os; os.makedirs(a.out, exist_ok=True)
np.savez_compressed(f"{a.out}/pipeline_{a.image}_{a.i}_{a.j}.npz", slide=slide.astype(np.float16),
                    crop_ij=np.array([a.i, a.j]), emb=emb, z=Z.T.reshape(128, 16, 16),
                    evr=pca.explained_variance_ratio_, markers=np.array(markers()))
print("n channels slide", slide.shape)
