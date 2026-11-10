"""Per-node reconstruction error over the 10,197 validation crops (test.h5) (szary, GPU).

``reference`` (the final model, PCA-128 on the 16x16 grid, zscore=False, no clip), per node:
    err_pca    ||x_hat - x||^2 / 128                 error in the space the model is trained in
    err_model  ||inv(x_hat) - emb||^2 / 768          total error in the original 768-d space
    err_trunc  ||inv(x) - emb||^2 / 768              PCA truncation floor (perfect decoder)
  plus per-crop error vs target norm and the explained-variance curves of all fitted PCAs
  -> figures/_cache/recon_error_81zasevr.npz

``run --run-dir DIR --tag TAG`` (any training run, rebuilt from its own .hydra/config.yaml with that
  run's centre crop, PCA, optional z-score and clip): per-node error in the model's training space and
  error binned by per-node target magnitude -> figures/_cache/recon_error_<TAG>.npz

Usage:
    uv run python scripts/thesis_figures/extract_recon_error.py reference
    uv run python scripts/thesis_figures/extract_recon_error.py run --run-dir /raid_.../runs/<run> --tag r1_cuogzab1
"""

import argparse
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import h5py
import joblib
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from figlib import paths
from src.data.components.graphs_datamodules import DenseGraphBatch, PCALayer
from src.downstream.encode import load_frozen_pigvae

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAGNITUDE_BINS = np.linspace(0, 3, 31)

for _name, _fn in [
    ("multiply", lambda x, y: int(x) * int(y)),
    ("divide", lambda x, y: int(x) // int(y)),
]:
    OmegaConf.register_new_resolver(_name, _fn, replace=True)


@torch.no_grad()
def reconstruct(gae, x: torch.Tensor) -> torch.Tensor:
    """Deterministic encode -> decode of node features [B, N, F]."""
    batch = DenseGraphBatch(
        node_features=x,
        edge_features=torch.empty(0, device=DEVICE),
        mask=torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=DEVICE),
    )
    z_nodes, z_global, *_ = gae.encode(batch, sample=False)
    return gae.decode(z_nodes, z_global, batch.mask).node_features.float()


def ring_index(g: int) -> np.ndarray:
    d = np.minimum(np.arange(g), g - 1 - np.arange(g))
    return np.minimum.outer(d, d)


def reference(batch_size: int, out: str) -> None:
    explained = {}
    for p in sorted(paths.CORDS_DATA.glob("pca_model_*_center_crop_*.pkl")):
        key = p.name.split("pca_model_")[1][:-4]
        explained[key] = joblib.load(p).explained_variance_ratio_
        print(key, "total explained variance", explained[key].sum())

    pca = joblib.load(paths.PCA_128)
    components = torch.tensor(pca.components_, dtype=torch.float32, device=DEVICE)  # [128, 768]
    mean = torch.tensor(pca.mean_, dtype=torch.float32, device=DEVICE)
    gae = load_frozen_pigvae(str(paths.REF_CKPT), paths.REF_EXPERIMENT).to(DEVICE)

    sums = {
        k: torch.zeros(256, dtype=torch.float64, device=DEVICE)
        for k in ["pca", "model", "trunc", "tgt"]
    }
    crop_err, crop_norm, n = [], [], 0
    with h5py.File(paths.CORDS_DATA / "test.h5", "r") as f:
        embeddings = f["embeddings"]
        for start in range(0, embeddings.shape[0], batch_size):
            emb = torch.from_numpy(embeddings[start : start + batch_size]).to(
                DEVICE
            )  # [B, 768, 16, 16]
            b = emb.shape[0]
            raw = emb.reshape(b, 768, 256).transpose(1, 2)  # [B, 256, 768], row-major nodes
            x = (raw - mean) @ components.T  # [B, 256, 128]
            xh = reconstruct(gae, x)
            e_pca = ((xh - x) ** 2).mean(-1)
            sums["pca"] += e_pca.sum(0).double()
            sums["model"] += (((xh @ components + mean) - raw) ** 2).mean(-1).sum(0).double()
            sums["trunc"] += (((x @ components + mean) - raw) ** 2).mean(-1).sum(0).double()
            sums["tgt"] += (x**2).mean(-1).sum(0).double()
            crop_err.append(e_pca.mean(1).cpu())
            crop_norm.append((x**2).mean((1, 2)).sqrt().cpu())
            n += b
            if (start // batch_size) % 20 == 0:
                print(f"{n}/{embeddings.shape[0]}", flush=True)

    per_node = {k: (v / n).cpu().numpy().reshape(16, 16) for k, v in sums.items()}
    print(
        f"mean err_pca {per_node['pca'].mean():.4f}  err_model {per_node['model'].mean():.4f}  "
        f"err_trunc {per_node['trunc'].mean():.4f}  -> truncation share "
        f"{100 * per_node['trunc'].mean() / per_node['model'].mean():.1f}%"
    )
    np.savez_compressed(
        out,
        err_pca=per_node["pca"],
        err_model=per_node["model"],
        err_trunc=per_node["trunc"],
        tgt_pow=per_node["tgt"],
        crop_err=torch.cat(crop_err).numpy(),
        crop_norm=torch.cat(crop_norm).numpy(),
        n=n,
        **{f"evr_{k}": v for k, v in explained.items()},
    )
    print("saved", out)


def load_run(run_dir: Path, no_final_norm: bool):
    """The run's graph autoencoder and data transform, rebuilt from its saved config."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    if no_final_norm:
        OmegaConf.update(
            cfg, "model.graph_ae.hparams.encoder.use_final_norm", False, force_add=True
        )
    gae = instantiate(cfg.model.graph_ae)
    state = torch.load(
        run_dir / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False
    )["state_dict"]
    state = {k[len("graph_ae.") :]: v for k, v in state.items() if k.startswith("graph_ae.")}
    missing, unexpected = gae.load_state_dict(state, strict=False)
    h = cfg.data.hparams
    g, k = int(h.center_crop_size), int(h.num_pca_components)
    print(
        f"grid {g} pca {k} zscore {h.zscore} clip {h.clip_range} | missing {len(missing)} unexpected {len(unexpected)}"
    )
    pca = PCALayer(
        str(paths.CORDS_DATA / f"pca_model_{k}_center_crop_{g}.pkl"),
        str(paths.CORDS_DATA / f"imc_statistics_{k}_center_crop_{g}.pt"),
        clip_range=h.clip_range,
        zscore=h.zscore,
    ).to(DEVICE)
    return gae.to(DEVICE).eval(), pca, g


def run(run_dir: str, tag: str, batch_size: int, no_final_norm: bool, out_dir: str) -> None:
    gae, pca, g = load_run(Path(run_dir), no_final_norm)
    n_nodes, offset = g * g, (16 - g) // 2
    err_sum = torch.zeros(n_nodes, dtype=torch.float64, device=DEVICE)
    tgt_sum = torch.zeros_like(err_sum)
    bin_err, bin_n, n = np.zeros(30), np.zeros(30), 0
    with h5py.File(paths.CORDS_DATA / "test.h5", "r") as f:
        embeddings = f["embeddings"]
        for start in range(0, embeddings.shape[0], batch_size):
            emb = torch.from_numpy(
                embeddings[start : start + batch_size][
                    :, :, offset : offset + g, offset : offset + g
                ]
            )
            emb = emb.to(DEVICE)
            b = emb.shape[0]
            x = pca(emb.reshape(b, 768, n_nodes).transpose(1, 2))
            e = ((reconstruct(gae, x) - x) ** 2).mean(-1)
            norm = (x**2).mean(-1).sqrt()
            err_sum += e.sum(0).double()
            tgt_sum += (x**2).mean(-1).sum(0).double()
            n += b
            idx = np.clip(np.digitize(norm.cpu().numpy().ravel(), MAGNITUDE_BINS) - 1, 0, 29)
            np.add.at(bin_err, idx, e.cpu().numpy().ravel())
            np.add.at(bin_n, idx, 1)

    err = (err_sum / n).cpu().numpy().reshape(g, g)
    tgt = (tgt_sum / n).cpu().numpy().reshape(g, g)
    rings = ring_index(g)
    corners = err[[0, 0, -1, -1], [0, -1, 0, -1]].mean() / err[rings == rings.max()].mean()
    print(
        f"[{tag}] mean {err.mean():.4f} | by ring:",
        " ".join(f"{err[rings == i].mean():.4f}" for i in range(rings.max() + 1)),
        f"| corner/centre {corners:.2f}",
    )
    out = Path(out_dir) / f"recon_error_{tag}.npz"
    np.savez_compressed(
        out,
        err=err,
        tgt=tgt,
        n=n,
        mag_bins=MAGNITUDE_BINS,
        mag_err=bin_err / np.maximum(bin_n, 1),
        mag_n=bin_n,
    )
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    ref = sub.add_parser("reference", help="the final model, three error decompositions")
    ref.add_argument("--batch-size", type=int, default=64)
    ref.add_argument("--out", default=str(paths.CACHE / f"recon_error_{paths.REF_WANDB_ID}.npz"))
    one = sub.add_parser("run", help="any training run dir, in its own training space")
    one.add_argument("--run-dir", required=True)
    one.add_argument("--tag", required=True, help="cache name suffix, e.g. r1_cuogzab1")
    one.add_argument("--batch-size", type=int, default=128)
    one.add_argument(
        "--no-final-norm",
        action="store_true",
        help="run was trained without the encoder final RMSNorm",
    )
    one.add_argument("--out-dir", default=str(paths.CACHE))
    a = ap.parse_args()
    if a.mode == "reference":
        reference(a.batch_size, a.out)
    else:
        run(a.run_dir, a.tag, a.batch_size, a.no_final_norm, a.out_dir)
