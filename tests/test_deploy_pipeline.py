import numpy as np
import pytest
import torch

from src.data.components.graphs_datamodules import DenseGraphBatch
from src.deploy.pipeline import (
    AbmilHead,
    IMCPipeline,
    InvalidImageError,
    project_nodes,
    validate_image,
)
from src.downstream.abmil.model import GatedABMIL

D = 16
# Fake panel with the default preview markers at their cords positions.
MARKERS = [f"m{i}" for i in range(43)]
MARKERS[34], MARKERS[13], MARKERS[19] = "panCK", "CD45", "DNA1"


class FakeImmuVis:
    """[B, 43, 128, 128] -> [B, 768, 16, 16] with a fixed random channel projection."""

    def __init__(self):
        self.w = torch.randn(768, 43, generator=torch.Generator().manual_seed(0))

    def __call__(self, x):
        pooled = torch.nn.functional.avg_pool2d(x, 8)  # [B, 43, 16, 16]
        return torch.einsum("oc,bchw->bohw", self.w, pooled)


class FakeGAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(128, D)

    def encode(self, batch: DenseGraphBatch, sample: bool = False):
        assert batch.node_features.shape[1:] == (256, 128)
        return None, self.lin(batch.node_features.mean(1))


def make_pipeline(num_classes: int = 2) -> IMCPipeline:
    torch.manual_seed(0)
    head = AbmilHead(
        model=GatedABMIL(D, 8, 1, num_classes),
        mu=torch.zeros(D),
        sd=torch.ones(D),
        classes=[f"class{i}" for i in range(num_classes)],
        label="DX.name",
        fold=3,
        fold_val_auroc=0.9,
    )
    return IMCPipeline(
        FakeImmuVis(),
        torch.randn(128, 768),
        torch.zeros(768),
        FakeGAE(),
        head,
        markers=MARKERS,
        manifest={"bundle_id": "test"},
        batch_size=2,
    )


def raw_image(h=300, w=270, channels_last=False, dtype=np.uint16):
    img = np.random.default_rng(0).poisson(10, size=(43, h, w)).astype(dtype)
    return np.moveaxis(img, 0, -1) if channels_last else img


def test_validate_image_accepts_channels_last_and_ints():
    out = validate_image(raw_image(channels_last=True))
    assert out.shape == (43, 300, 270) and out.dtype == np.float32


@pytest.mark.parametrize(
    "img, msg",
    [
        (np.zeros((43, 300)), "3-D"),
        (np.zeros((40, 300, 300)), "43 channels"),
        (np.zeros((43, 100, 300)), "smaller than one 128 px crop"),
        (np.zeros((43, 128, 5000)), "exceeds"),
        (np.full((43, 128, 128), np.nan), "NaN"),
    ],
)
def test_validate_image_rejects(img, msg):
    with pytest.raises(InvalidImageError, match=msg):
        validate_image(img)


def test_project_nodes_is_row_major_pca():
    latent = torch.randn(2, 768, 16, 16)
    comps, mean = torch.randn(128, 768), torch.randn(768)
    out = project_nodes(latent, comps, mean)
    assert out.shape == (2, 256, 128)
    # node 17 = grid cell (row 1, col 1)
    assert torch.allclose(out[1, 17], (latent[1, :, 1, 1] - mean) @ comps.T, atol=1e-3)


def test_embed_crops_and_order():
    emb = make_pipeline().embed(raw_image())
    assert emb.coords == [(0, 0), (0, 128), (128, 0), (128, 128)]
    assert emb.z_global.shape == (4, D)


def test_predict_binary_response():
    out = make_pipeline().predict(raw_image(), ig_steps=4, include_embeddings=True)
    assert out["label"] == "DX.name"
    assert set(out["probabilities"]) == {"class0", "class1"}
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-6
    assert out["prediction"] == max(out["probabilities"], key=out["probabilities"].get)
    assert isinstance(out["logit"], float)
    assert out["image"] == {"height": 300, "width": 270, "tile_size": 128, "n_crops": 4}
    assert abs(sum(c["attention"] for c in out["crops"]) - 1) < 1e-5
    assert all(len(c["z_global"]) == D and c["ig_importance"] is not None for c in out["crops"])
    assert out["model"] == {"label": "DX.name", "fold": 3, "fold_val_auroc": 0.9, "bundle": "test"}


def test_predict_without_ig_or_embeddings():
    out = make_pipeline().predict(raw_image(), ig_steps=0)
    assert all(c["ig_importance"] is None and c["z_global"] is None for c in out["crops"])


def test_predict_multiclass():
    out = make_pipeline(num_classes=3).predict(raw_image(), ig_steps=4)
    assert len(out["probabilities"]) == 3 and isinstance(out["logit"], list)


def test_ig_completeness_binary():
    """IG over the bag sums to logit(bag) - logit(zero bag) for the single-logit head."""
    pipe = make_pipeline()
    out = pipe.predict(raw_image(), ig_steps=200)
    total = sum(c["ig_importance"] for c in out["crops"])
    zero = pipe.head.model(torch.zeros(1, 4, D))[0].item()
    assert abs(total - (out["logit"] - zero)) < 1e-2


def test_info_lists_markers():
    info = make_pipeline().info()
    assert info["n_markers"] == 43 and info["classes"] == ["class0", "class1"]
