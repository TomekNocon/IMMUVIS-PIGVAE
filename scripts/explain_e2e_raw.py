"""End-to-end raw-image explainability (two views per DX image).

For one clinical image it produces, on the REAL cords tissue:
  (1) ABMIL decision mosaic -- each crop stitched at its tissue coords, coloured by its
      Integrated-Gradients contribution to the DX logit; the decisive crop is outlined.
      => "which crop drove the diagnosis", on the raw image.
  (2) Representation map on the decisive crop -- that crop's raw markers go through the
      ImmuVis-768-UN-475 encoder -> PCA -> PIGVAE, and encoder-IG of z_global is overlaid on the
      false-colour crop. => "where on the crop the z_global the decision uses comes from".

Chains the trained ABMIL head + PIGVAE (81zasevr) + ImmuVis encoder (submodule). Runs on szary.
"""

import operator
import os

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("multiply", lambda x, y: operator.mul(int(x), int(y)), replace=True)
OmegaConf.register_new_resolver("divide", lambda x, y: int(x) // int(y), replace=True)

from src.downstream.abmil.data import _load_memmap
from src.downstream.abmil.lit import AbmilLitModule
from src.explainability import imc_viz
from src.explainability import immuvis_encoder as ie
from src.explainability.abmil_attribution import abmil_attention, abmil_ig, crop_importance
from src.explainability.encoder_attribution import encoder_ig


def _balance_meta_df(meta_df, min_class_freq=0.05):
    img_labels = meta_df.groupby("img_path", sort=False)["feature_value"].first()
    img_labels = img_labels[~pd.isna(img_labels)].astype(str)
    if len(img_labels) == 0:
        return meta_df.iloc[0:0]
    classes, counts = np.unique(img_labels.values, return_counts=True)
    keep = set(classes[(counts / counts.sum()) >= float(min_class_freq)])
    keep_imgs = img_labels[img_labels.isin(keep)].index
    return meta_df[meta_df["img_path"].isin(keep_imgs)]


def build_bags(meta_df, class_to_idx):
    """Per image: (crops[S,D], label, coords[S,2], img_path)."""
    out = []
    for img, g in meta_df.groupby("img_path", sort=False):
        raw = g["feature_value"].iloc[0]
        lab = class_to_idx.get(str(raw))
        if lab is None or pd.isna(raw):
            continue
        rows, crd = [], []
        for ef, gg in g.groupby("embeddings_file"):
            rows.append(_load_memmap(ef)[gg["embedding_idx"].values])
            crd.append(gg[["coords0", "coords1"]].values)
        out.append((np.concatenate(rows, 0), lab, np.concatenate(crd, 0), str(img)))
    return out


def _resolve_tiff(img_path, tiff_root, dataset):
    """Find the whole-image tiff under <tiff_root>/<split>/<dataset>/imgs/ for this img_path."""
    base = os.path.basename(str(img_path))
    if not base.lower().endswith((".tif", ".tiff")):
        base = base + ".tiff"
    for split in ("train", "test"):
        p = os.path.join(tiff_root, split, dataset, "imgs", base)
        if os.path.exists(p):
            return p
    return None


def _preprocess_for_encoder(raw_chw):
    """arcsinh(x/5) -> butterworth(0.2) -> clip(0,4)/4  (the generate_patches recipe for cords)."""
    x = np.arcsinh(np.asarray(raw_chw, dtype=np.float32) / 5.0)
    x = imc_viz.butterworth_denoise(x, 0.2)
    return np.clip(x, 0.0, 4.0) / 4.0


@hydra.main(version_base="1.3", config_path="../configs", config_name="explain_e2e")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    dev = "cpu"

    # ── models ──
    print("[e2e] loading PIGVAE ...")
    pig = hydra.utils.instantiate(cfg.model)
    ck = torch.load(cfg.pigvae_ckpt, map_location="cpu", weights_only=False)
    miss, unexp = pig.load_state_dict(ck["state_dict"], strict=False)
    pig.eval(); gae = pig.graph_ae
    print(f"      PIGVAE loaded (epoch {ck.get('epoch','?')}) missing={len(miss)} unexpected={len(unexp)}")

    print("[e2e] datamodule (pca_layer) ...")
    dm = hydra.utils.instantiate(cfg.data); dm.setup("test")
    pca_layer = dm.pca_layer

    print("[e2e] ImmuVis-768-UN-475 ...")
    imm, (m2, u2) = ie.load_immuvis(device=dev)
    cids, mnames = ie.dataset_channel_ids(cfg.dataset)
    print(f"      ImmuVis loaded missing={len(m2)} unexpected={len(u2)}; {len(mnames)} markers")

    print("[e2e] ABMIL head ...")
    abmil = AbmilLitModule.load_from_checkpoint(cfg.abmil_ckpt, map_location="cpu").model.eval()

    # ── meta -> binary DX bags ──
    frames = []
    for split in ("train", "test"):
        p = os.path.join(cfg.meta_dir, f"{cfg.dataset}_{split}_metadata.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    meta = pd.concat(frames, ignore_index=True)
    if "feature" in meta.columns:
        meta = meta[meta["feature"] == cfg.feature].copy()
    meta = meta[meta["img_path"].isin(meta.dropna(subset=["feature_value"])["img_path"].unique())]
    meta = _balance_meta_df(meta, cfg.min_class_freq)
    classes = sorted(meta["feature_value"].astype(str).unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[e2e] classes={class_to_idx}")

    bags = build_bags(meta, class_to_idx)
    allc = np.concatenate([b[0] for b in bags], 0)
    mu, sd = allc.mean(0), allc.std(0) + 1e-6

    comp = list(cfg.composite_channels)
    pctl = tuple(cfg.pctl)
    idx_to_cls = {v: k for k, v in class_to_idx.items()}
    short = {0: "adeno", 1: "squam"}

    # ── phase 1: score every bag (cheap forward), pick confident-correct ones per class ──
    scored = []
    for bidx, (crops, lab, coords, img_path) in enumerate(bags):
        if _resolve_tiff(img_path, cfg.tiff_root, cfg.dataset) is None:
            continue
        x = torch.from_numpy(((crops - mu) / sd).astype("float32")).unsqueeze(0)
        with torch.no_grad():
            logit = float(abmil(x)[0].item())
        correct = (logit > 0) == (lab == 1)
        scored.append((bidx, lab, logit, correct))
    picks = []
    for cl in sorted(class_to_idx.values()):
        cands = [s for s in scored if s[1] == cl]
        cands.sort(key=lambda s: (s[3], abs(s[2])), reverse=True)  # correct first, then confident
        picks += cands[: int(cfg.images_per_class)]
    print(f"[e2e] picked {len(picks)} images: "
          + ", ".join(f"{idx_to_cls[l][:5]}(logit{lo:+.2f}{'ok' if c else 'X'})" for _, l, lo, c in picks))

    dims = None
    for n, (bidx, lab, _lg, _ok) in enumerate(picks):
        crops, lab, coords, img_path = bags[bidx]
        tiff = _resolve_tiff(img_path, cfg.tiff_root, cfg.dataset)
        cls = short.get(lab, str(lab))

        # ABMIL decision + per-crop IG
        x = torch.from_numpy(((crops - mu) / sd).astype("float32")).unsqueeze(0)
        attr = abmil_ig(abmil, x, target=0, n_steps=int(cfg.n_steps))
        imp = crop_importance(attr)[0].numpy()
        logit = float(abmil(x)[0].item())
        # IG is signed toward the Squamous logit; orient toward the PREDICTED class so that, for
        # BOTH classes, high (red) = the crop supports the model's call and argmax = the crop that
        # most supports it (not the strongest counter-evidence).
        imp = imp * (1.0 if logit > 0 else -1.0)
        pred = short.get(int(logit > 0), "?")

        # raw tiff -> per-crop false-colour
        raw = imc_viz.load_imc_image(tiff, arcsinh_scale=None)
        patches, pcoords = imc_viz.get_all_patches(raw, 128)
        topleft = {(int(a[0]), int(a[1])): k for k, (a, _b) in enumerate(pcoords)}
        crops_rgb, dens = [], np.full(len(coords), -1.0, dtype=np.float32)
        for k in range(len(coords)):
            key = (int(coords[k, 0]), int(coords[k, 1]))
            pk = topleft.get(key, topleft.get((key[1], key[0])))
            if pk is None:
                crops_rgb.append(np.zeros((128, 128, 3), np.float32)); continue
            arc = np.arcsinh(patches[pk] / 5.0)
            dens[k] = float(arc.mean())   # RAW tissue density (false-colour mean is misleading)
            crops_rgb.append(imc_viz.compose_false_colour(
                arc, imc_viz._resolve_channels(mnames, comp),
                imc_viz.DEFAULT_COLOURS[:len(comp)], pctl=pctl))
        # decisive = strongest supporter among REAL-tissue crops (raw density well above zero).
        # NB: the single top raw supporter can be a near-EMPTY crop -- a model shortcut we keep
        # visible in the heatmap but do not point the box at.
        valid = dens[dens >= 0]
        thr = max(0.1, 0.4 * float(np.median(valid))) if len(valid) else 0.1
        has_tissue = dens > thr
        masked = np.where(has_tissue, imp, -np.inf)
        decisive = int(np.argmax(masked if np.isfinite(masked).any() else imp))
        top_raw = int(np.argmax(np.where(dens >= 0, imp, -np.inf)))
        print(f"[e2e] {os.path.basename(tiff)} true={cls} pred={pred} logit={logit:+.3f} "
              f"crops={len(crops)} decisive(tissue)={decisive} dens={dens[decisive]:.3f} "
              f"| top-raw-supporter={top_raw} dens={dens[top_raw]:.3f}")

        # ── IMAGE 1: ABMIL decision heatmap ──
        tag = f"{n}_{cls}_{os.path.splitext(os.path.basename(tiff))[0]}"
        out1 = os.path.join(cfg.out_dir, f"{tag}_decision.png")
        imc_viz.stitch_composite_importance(
            crops_rgb, coords, imp, out1, highlight=decisive,
            title=f"{os.path.basename(tiff)} · true={cls} pred={pred} (logit {logit:+.2f})")

        # ── IMAGE 2: representation GRID over many z_global dims, on the decisive crop ──
        dkey = (int(coords[decisive, 0]), int(coords[decisive, 1]))
        pk = topleft.get(dkey, topleft.get((dkey[1], dkey[0])))
        if pk is not None:
            pre = _preprocess_for_encoder(patches[pk])
            xin = torch.from_numpy(pre).unsqueeze(0).float()
            with torch.no_grad():
                lat = ie.encode(imm, xin, cids.unsqueeze(0))
                nf = pca_layer(lat.reshape(1, lat.shape[1], -1).transpose(1, 2)).float()  # [1,256,128]
            if dims is None:
                from src.explainability.encoder_attribution import _encode_zglobal
                zdim = _encode_zglobal(gae, nf).shape[-1]
                dims = list(np.linspace(0, zdim - 1, int(cfg.repr_dims)).astype(int))
            maps = []
            for d in dims:
                a = encoder_ig(gae, nf, target=int(d), n_steps=int(cfg.repr_n_steps))
                maps.append(a.abs().sum(-1)[0].reshape(16, 16).numpy())
            out2 = os.path.join(cfg.out_dir, f"{tag}_repr_dims.png")
            imc_viz.plot_repr_dim_grid(
                crops_rgb[decisive], maps, dims, out2,
                title=f"{os.path.basename(tiff)} decisive crop · z_global dims")
            print(f"      wrote {out1}\n      wrote {out2}")

    print(f"[e2e] done -> {os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    main()
