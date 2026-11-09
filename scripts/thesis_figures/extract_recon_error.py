"""Per-node reconstruction error of the trained PIGVAE over the validation crops (szary, GPU).

For every crop of test.h5 (the 10,197-crop validation split) in training node order:
  x      = PCA-128(emb)                         (the model's input/target; zscore=False, no clip)
  x_hat  = decode(encode(x)), deterministic
Per node we accumulate:
  err_pca   = ||x_hat - x||^2 / 128              (error in the space the model is trained in)
  err_model = ||inv(x_hat) - emb||^2 / 768       (total error in the original 768-d feature space)
  err_trunc = ||inv(x) - emb||^2 / 768           (PCA truncation floor: perfect decoder, same PCA)
plus per-crop target norm vs error, and the explained-variance curves of the fitted PCAs.
"""
import argparse, glob, sys
import h5py, joblib, numpy as np, torch
sys.path.insert(0, ".")
from src.downstream.encode import load_frozen_pigvae
from src.data.components.graphs_datamodules import DenseGraphBatch

D = "/raid_encrypted/immucan/embeddings/tnocon/data/IMC/cords"
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/raid_encrypted/immucan/embeddings/tnocon/logs/train/runs/2026-07-31_10-29-22/checkpoints/last.ckpt")
ap.add_argument("--experiment", default="pigvae")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--out", default="thesis_figures/_cache/recon_error_81zasevr.npz")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev)

evr = {}
for p in sorted(glob.glob(f"{D}/pca_model_*_center_crop_*.pkl")):
    k = p.split("pca_model_")[1][:-4]
    evr[k] = joblib.load(p).explained_variance_ratio_
    print(k, "total", evr[k].sum())

pca = joblib.load(f"{D}/pca_model_128_center_crop_16.pkl")
V = torch.tensor(pca.components_, dtype=torch.float32, device=dev)     # [128, 768]
m = torch.tensor(pca.mean_, dtype=torch.float32, device=dev)
gae = load_frozen_pigvae(a.ckpt, a.experiment).to(dev)

sums = {k: torch.zeros(256, dtype=torch.float64, device=dev) for k in ["pca", "model", "trunc", "tgt"]}
crop_err, crop_norm, n = [], [], 0
with h5py.File(f"{D}/test.h5", "r") as f:
    E = f["embeddings"]; total = E.shape[0]
    for s in range(0, total, a.bs):
        emb = torch.from_numpy(E[s:s + a.bs]).to(dev)                          # [B, 768, 16, 16]
        B = emb.shape[0]
        raw = emb.reshape(B, 768, 256).transpose(1, 2)                         # [B, 256, 768] row-major
        x = (raw - m) @ V.T                                                    # [B, 256, 128]
        batch = DenseGraphBatch(node_features=x, edge_features=torch.empty(0, device=dev),
                                mask=torch.ones(B, 256, dtype=torch.bool, device=dev))
        with torch.no_grad():
            zn, zg, *_ = gae.encode(batch, sample=False)
            xh = gae.decode(zn, zg, batch.mask).node_features.float()
        e_pca = ((xh - x) ** 2).mean(-1)
        e_mod = (((xh @ V + m) - raw) ** 2).mean(-1)
        e_tr = (((x @ V + m) - raw) ** 2).mean(-1)
        sums["pca"] += e_pca.sum(0).double(); sums["model"] += e_mod.sum(0).double()
        sums["trunc"] += e_tr.sum(0).double(); sums["tgt"] += (x ** 2).mean(-1).sum(0).double()
        crop_err.append(e_pca.mean(1).cpu()); crop_norm.append((x ** 2).mean((1, 2)).sqrt().cpu())
        n += B
        if (s // a.bs) % 20 == 0: print(f"{n}/{total}", flush=True)

out = {k: (v / n).cpu().numpy().reshape(16, 16) for k, v in sums.items()}
print("mean err_pca %.4f  err_model %.4f  err_trunc %.4f  -> trunc share %.1f%%" % (
    out["pca"].mean(), out["model"].mean(), out["trunc"].mean(), 100 * out["trunc"].mean() / out["model"].mean()))
np.savez_compressed(a.out, err_pca=out["pca"], err_model=out["model"], err_trunc=out["trunc"], tgt_pow=out["tgt"],
                    crop_err=torch.cat(crop_err).numpy(), crop_norm=torch.cat(crop_norm).numpy(), n=n,
                    **{f"evr_{k}": v for k, v in evr.items()})
print("saved", a.out)
