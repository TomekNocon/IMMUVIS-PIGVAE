"""Per-node reconstruction error for any training run dir (rebuilt from its own .hydra/config.yaml).

Applies exactly that run's data transform: centre crop of the 16x16 map, PCA-k of that crop size,
optional z-score with the saved statistics and clipping (PCALayer). Error is measured in the
model's training space (what the loss sees), per node, averaged over all test.h5 crops.
Also reports error vs per-node target magnitude, binned, for the magnitude claim.
"""
import rootutils
import argparse
import h5py, numpy as np, torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from src.data.components.graphs_datamodules import DenseGraphBatch, PCALayer

D = "/raid_encrypted/immucan/embeddings/tnocon/data/IMC/cords"
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--no-final-norm", action="store_true", help="run trained without the encoder final RMSNorm")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
for name, fn in [("multiply", lambda x, y: int(x) * int(y)), ("divide", lambda x, y: int(x) // int(y))]:
    try: OmegaConf.register_new_resolver(name, fn)
    except Exception: pass
cfg = OmegaConf.load(f"{a.run}/.hydra/config.yaml")
h = cfg.data.hparams
g, k = int(h.center_crop_size), int(h.num_pca_components)
if a.no_final_norm:
    OmegaConf.update(cfg, "model.graph_ae.hparams.encoder.use_final_norm", False, force_add=True)
gae = instantiate(cfg.model.graph_ae)
sd = torch.load(f"{a.run}/checkpoints/last.ckpt", map_location="cpu", weights_only=False)["state_dict"]
sd = {kk[len("graph_ae."):]: v for kk, v in sd.items() if kk.startswith("graph_ae.")}
missing, unexpected = gae.load_state_dict(sd, strict=False)
print(f"[{a.tag}] grid {g} pca {k} zscore {h.zscore} clip {h.clip_range} | missing {len(missing)} unexpected {len(unexpected)}")
if missing: print("  missing:", missing[:8])
if unexpected: print("  unexpected:", unexpected[:8])
gae = gae.to(dev).eval()
pca = PCALayer(f"{D}/pca_model_{k}_center_crop_{g}.pkl", f"{D}/imc_statistics_{k}_center_crop_{g}.pt",
               clip_range=h.clip_range, zscore=h.zscore).to(dev)

N = g * g; t0 = (16 - g) // 2
err_sum = torch.zeros(N, dtype=torch.float64, device=dev); tgt_sum = torch.zeros_like(err_sum)
bins = np.linspace(0, 3, 31); hist_err = np.zeros(30); hist_n = np.zeros(30); n = 0
with h5py.File(f"{D}/test.h5", "r") as f:
    E = f["embeddings"]
    for s in range(0, E.shape[0], a.bs):
        emb = torch.from_numpy(E[s:s + a.bs][:, :, t0:t0 + g, t0:t0 + g]).to(dev)
        B = emb.shape[0]
        x = pca(emb.reshape(B, 768, N).transpose(1, 2))
        batch = DenseGraphBatch(node_features=x, edge_features=torch.empty(0, device=dev),
                                mask=torch.ones(B, N, dtype=torch.bool, device=dev))
        with torch.no_grad():
            zn, zg, *_ = gae.encode(batch, sample=False)
            xh = gae.decode(zn, zg, batch.mask).node_features.float()
        e = ((xh - x) ** 2).mean(-1); tn = (x ** 2).mean(-1).sqrt()
        err_sum += e.sum(0).double(); tgt_sum += (x ** 2).mean(-1).sum(0).double(); n += B
        idx = np.clip(np.digitize(tn.cpu().numpy().ravel(), bins) - 1, 0, 29)
        np.add.at(hist_err, idx, e.cpu().numpy().ravel()); np.add.at(hist_n, idx, 1)
err = (err_sum / n).cpu().numpy().reshape(g, g); tgt = (tgt_sum / n).cpu().numpy().reshape(g, g)
r = np.minimum.outer(np.minimum(np.arange(g), g - 1 - np.arange(g)), np.minimum(np.arange(g), g - 1 - np.arange(g)))
print(f"[{a.tag}] mean {err.mean():.4f} | by ring:", " ".join(f"{err[r == i].mean():.4f}" for i in range(r.max() + 1)),
      f"| corner/centre {err[[0, 0, -1, -1], [0, -1, 0, -1]].mean() / err[r == r.max()].mean():.2f}")
np.savez_compressed(f"figures/_cache/recon_error_{a.tag}.npz", err=err, tgt=tgt, n=n,
                    mag_bins=bins, mag_err=hist_err / np.maximum(hist_n, 1), mag_n=hist_n)
