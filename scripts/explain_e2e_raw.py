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
    done = 0
    for bidx, (crops, lab, coords, img_path) in enumerate(bags):
        if done >= int(cfg.num_images):
            break
        tiff = _resolve_tiff(img_path, cfg.tiff_root, cfg.dataset)
        if tiff is None:
            print(f"[e2e] img {img_path}: no tiff found, skip")
            continue

        # ABMIL decision + per-crop IG on the (zscored) bag
        x = torch.from_numpy(((crops - mu) / sd).astype("float32")).unsqueeze(0)
        attn = abmil_attention(abmil, x)[0, :, 0]
        attr = abmil_ig(abmil, x, target=0, n_steps=int(cfg.n_steps))
        imp = crop_importance(attr)[0].numpy()          # [S]
        logit = float(abmil(x)[0].item())
        decisive = int(np.argmax(imp))
        print(f"[e2e] img={os.path.basename(tiff)} label={lab} logit={logit:+.3f} "
              f"crops={len(crops)} decisive_crop={decisive} imp={imp[decisive]:+.3f}")

        # raw tiff -> per-crop false-colour + coord match
        raw = imc_viz.load_imc_image(tiff, arcsinh_scale=None)          # raw counts [43,H,W]
        patches, pcoords = imc_viz.get_all_patches(raw, 128)
        topleft = {(int(a[0]), int(a[1])): k for k, (a, _b) in enumerate(pcoords)}
        crops_rgb, matched = [], 0
        for k in range(len(coords)):
            key = (int(coords[k, 0]), int(coords[k, 1]))
            pk = topleft.get(key, topleft.get((key[1], key[0])))   # try swapped order too
            if pk is None:
                crops_rgb.append(np.zeros((128, 128, 3), np.float32)); continue
            matched += 1
            arc = np.arcsinh(patches[pk] / 5.0)
            crops_rgb.append(imc_viz.compose_false_colour(arc, imc_viz._resolve_channels(mnames, comp),
                                                          imc_viz.DEFAULT_COLOURS[:len(comp)], pctl=pctl))
        print(f"      coord match: {matched}/{len(coords)} crops")

        # ── IMAGE 1: ABMIL decision mosaic on raw tissue ──
        out1 = os.path.join(cfg.out_dir, f"img{done}_abmil_decision.png")
        imc_viz.stitch_composite_importance(
            crops_rgb, coords, imp, out1, highlight=decisive,
            title=f"{os.path.basename(tiff)} · ABMIL decision (label {lab}, logit {logit:+.2f})")

        # ── IMAGE 2: z_global representation on the decisive raw crop ──
        dkey = (int(coords[decisive, 0]), int(coords[decisive, 1]))
        pk = topleft.get(dkey, topleft.get((dkey[1], dkey[0])))
        if pk is not None:
            pre = _preprocess_for_encoder(patches[pk])                        # [43,128,128] in [0,1]
            xin = torch.from_numpy(pre).unsqueeze(0).float()                  # [1,43,128,128]
            with torch.no_grad():
                lat = ie.encode(imm, xin, cids.unsqueeze(0))                  # [1,768,16,16]
                nf768 = lat.reshape(1, lat.shape[1], -1).transpose(1, 2)      # [1,256,768]
                nf = pca_layer(nf768)                                         # [1,256,128]
            a = encoder_ig(gae, nf.float(), target=int(cfg.target), n_steps=int(cfg.n_steps))
            node_imp = a.abs().sum(-1)[0].reshape(16, 16).numpy()            # [16,16]
            out2 = os.path.join(cfg.out_dir, f"img{done}_repr_decisive_crop.png")
            imc_viz.overlay_heatmap_on_rgb(
                crops_rgb[decisive], node_imp, out2, pixel_size_um=cfg.pixel_size_um,
                title=f"decisive crop · z_global[{cfg.target}] origin")
            print(f"      wrote {out1}\n      wrote {out2}")
        done += 1

    print(f"[e2e] done -> {os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    main()
