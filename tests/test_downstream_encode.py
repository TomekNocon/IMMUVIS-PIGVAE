import torch

def _build_tiny_graph_ae(tmp_path):
    """Build a graph_ae from the real model config so encode() is exercised, save a ckpt.

    Stamps a known, distinctive value into one specific `graph_ae` parameter
    before saving, then mutates the live parameter afterwards, so a test that
    later observes the marker value on a freshly-loaded module can only be
    explained by `load_frozen_pigvae` actually reading weights back off disk
    (as opposed to e.g. silently keeping a randomly-initialized module).
    """
    from src.downstream.encode import _build_pl_module  # helper Task A1 exposes
    pl = _build_pl_module(experiment="vae16_fb0p0", paths_name="szary")
    param_name, param = next(iter(pl.graph_ae.named_parameters()))
    with torch.no_grad():
        param.fill_(12345.0)
    marker_value = param.detach().clone()
    ckpt = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": pl.state_dict()}, ckpt)
    # Mutate the in-memory parameter after saving: the checkpoint on disk keeps
    # the marker value regardless of what happens to `pl` afterwards.
    with torch.no_grad():
        param.fill_(-1.0)
    return str(ckpt), param_name, marker_value

def test_load_frozen_pigvae_returns_eval_graph_ae(tmp_path):
    from src.downstream.encode import load_frozen_pigvae
    ckpt, param_name, marker_value = _build_tiny_graph_ae(tmp_path)
    gae = load_frozen_pigvae(ckpt, experiment="vae16_fb0p0")
    assert not gae.training                                        # eval mode
    assert all(not p.requires_grad for p in gae.parameters())      # weights frozen
    assert any(p.numel() > 0 for p in gae.parameters())            # weights present

    # Core promise of load_frozen_pigvae: weights were actually loaded from
    # ckpt_path, not just a freshly/randomly-initialized module of the right shape.
    loaded_param = dict(gae.named_parameters())[param_name]
    assert torch.equal(loaded_param, marker_value)
    # encode runs: 2 graphs, 256 nodes, 128 pca dims -> z_global [2, D]
    nf = torch.randn(2, 256, 128)
    from src.data.components.graphs_datamodules import DenseGraphBatch
    # A full 16x16 grid has no padding, i.e. an all-True mask (mask=None crashes:
    # GraphEncoder.add_emb_node_and_feature does F.pad(mask, ...) unconditionally
    # in CLS mode / use_pma=False, which this experiment config uses).
    mask = torch.ones(2, 256, dtype=torch.bool)
    batch = DenseGraphBatch(node_features=nf, edge_features=torch.empty(0), mask=mask)
    with torch.no_grad():
        z_nodes, z_global, *_ = gae.encode(batch, sample=False)
    assert z_global.shape[0] == 2 and z_global.dim() == 2


def test_patch_to_nodes_order_and_pca_shape():
    import numpy as np, torch
    from src.downstream.encode import patch_to_nodes
    patch = np.arange(768 * 16 * 16, dtype=np.float32).reshape(768, 16, 16)
    nodes = patch_to_nodes(patch)                 # (256, 768)
    assert nodes.shape == (256, 768)
    # node n = grid cell (row=n//16, col=n%16), all 768 channels of that cell
    assert torch.allclose(nodes[0], torch.from_numpy(patch[:, 0, 0]))
    assert torch.allclose(nodes[17], torch.from_numpy(patch[:, 1, 1]))


def test_encode_patches_deterministic_and_batch_agnostic(tmp_path):
    import numpy as np, torch
    from src.downstream.encode import load_frozen_pigvae, build_pca_layer, encode_patches, _build_pl_module
    # reuse the tiny ckpt + a stub PCALayer that just linearly maps 768->128
    class StubPCA:
        def __call__(self, x): return x[..., :128]
    pl = _build_pl_module("vae16_fb0p0"); ckpt = tmp_path/"c.ckpt"; torch.save({"state_dict": pl.state_dict()}, ckpt)
    gae = load_frozen_pigvae(str(ckpt), "vae16_fb0p0")
    patches = np.random.randn(5, 768, 16, 16).astype("float32")
    z1 = encode_patches(gae, StubPCA(), patches, device="cpu")
    z2 = encode_patches(gae, StubPCA(), patches, device="cpu")
    assert z1.shape[0] == 5
    assert np.allclose(z1, z2)                       # deterministic (sample=False)
    z_one = encode_patches(gae, StubPCA(), patches[:1], device="cpu")
    assert np.allclose(z_one[0], z1[0], atol=1e-5)   # batch-size independent

def test_memmap_writer_roundtrip(tmp_path):
    import numpy as np
    from src.downstream.memmap_writer import MemmapWriter
    p = str(tmp_path / "emb.npy")
    w = MemmapWriter(p, n_rows=10, dim=4)
    w.write(0, np.ones((3, 4), "float32"))
    w.write(3, np.full((7, 4), 2.0, "float32"))
    w.close()
    a = np.load(p, mmap_mode="r")
    assert a.shape == (10, 4) and a[0, 0] == 1.0 and a[9, 0] == 2.0


def test_encode_h5_alignment(tmp_path):
    import h5py, numpy as np, pandas as pd, torch
    from src.downstream.encode import encode_h5, load_frozen_pigvae, _build_pl_module
    class StubPCA:
        def __call__(self, x): return x[..., :128]
    h5 = tmp_path / "mini.h5"
    N = 7
    with h5py.File(h5, "w") as f:
        f["embeddings"] = np.random.randn(N, 768, 16, 16).astype("float32")
        f["paths"] = np.array([f"img{i//3}.tiff" for i in range(N)], dtype=object)
        f["positions"] = np.random.rand(N, 4).astype("float32")
    pl = _build_pl_module("vae16_fb0p0"); ck = tmp_path/"c.ckpt"; torch.save({"state_dict": pl.state_dict()}, ck)
    gae = load_frozen_pigvae(str(ck), "vae16_fb0p0")
    emb, meta = str(tmp_path/"out_embeddings.npy"), str(tmp_path/"out_metadata.csv")
    encode_h5(str(h5), gae, StubPCA(), emb, meta, device="cpu", batch_size=3)
    a = np.load(emb, mmap_mode="r"); df = pd.read_csv(meta)
    assert a.shape[0] == N and len(df) == N
    assert list(df["embedding_idx"]) == list(range(N))
    assert df["img_path"].iloc[0] == "img0.tiff" and df["img_path"].iloc[6] == "img2.tiff"
