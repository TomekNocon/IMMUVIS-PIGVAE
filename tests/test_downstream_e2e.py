"""End-to-end integration test: encode_h5 -> build_image_bags -> run_cv.

Exercises the full downstream chain on synthetic, tiny data (CPU only):
a synthetic h5 of raw IMC patches is encoded with a frozen (freshly built,
untrained) PIGVAE checkpoint + a stub PCA, the resulting metadata is joined
with a synthetic clinical label by `img_path`, grouped into per-image bags,
and run through stratified k-fold CV. This is a WIRING test -- it asserts
the pipeline runs clean end-to-end, not that it achieves any accuracy (the
labels are synthetic/near-random).
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import torch


class StubPCA:
    def __call__(self, x):
        return x[..., :128]


def test_e2e_encode_to_abmil(tmp_path):
    from src.downstream.encode import encode_h5, load_frozen_pigvae, _build_pl_module
    from src.downstream.abmil.data import build_image_bags
    from src.downstream.abmil.cv import run_cv

    # --- Step 1: synthetic h5 -- 12 patches, 4 unique images x 3 crops each ---
    n_images = 4
    crops_per_image = 3
    n = n_images * crops_per_image  # 12
    h5_path = tmp_path / "mini.h5"
    with h5py.File(h5_path, "w") as f:
        f["embeddings"] = np.random.randn(n, 768, 16, 16).astype("float32")
        paths = np.array(
            [f"img{i}" for i in range(n_images) for _ in range(crops_per_image)],
            dtype=object,
        )
        f["paths"] = paths
        f["positions"] = np.random.rand(n, 4).astype("float32")

    # --- Step 2: tiny (untrained) checkpoint + stub PCA, encode_h5 with ragged batches ---
    pl_module = _build_pl_module("vae16_fb0p0")
    ckpt_path = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": pl_module.state_dict()}, ckpt_path)
    gae = load_frozen_pigvae(str(ckpt_path), "vae16_fb0p0")

    out_emb = tmp_path / "e.npy"
    out_meta = tmp_path / "m.csv"
    # batch_size=5 does not divide n=12 -> exercises ragged batches
    encode_h5(str(h5_path), gae, StubPCA(), str(out_emb), str(out_meta), device="cpu", batch_size=5)

    meta_df = pd.read_csv(out_meta)
    assert len(meta_df) == n

    # --- Step 3: join a synthetic clinical label, 2 images class "A" + 2 class "B" ---
    label_by_img = {"img0": "A", "img1": "A", "img2": "B", "img3": "B"}
    meta_df["feature_value"] = meta_df["img_path"].map(label_by_img)

    # --- Step 4: build per-image bags ---
    bags, labels = build_image_bags(meta_df, {"A": 0, "B": 1})
    assert len(bags) == n_images
    for b in bags:
        assert b.shape[0] == crops_per_image

    # --- Step 5: run stratified 2-fold CV ---
    cfg = SimpleNamespace(
        num_folds=2, hidden_dim=8, num_heads=1, num_epochs=3, patience=2,
        lr=1e-2, batch_size=2, zscore="cv_train", results_dir=str(tmp_path),
    )
    out = run_cv(bags, labels, num_classes=2, cfg=cfg)

    # --- Step 6: assert the pipeline ran clean (not a performance assertion) ---
    assert len(bags) == 4
    assert out["num_folds_used"] == 2
    assert len(out["fold_accuracy"]) == 2
    assert 0.0 <= out["oof_accuracy"] <= 1.0
