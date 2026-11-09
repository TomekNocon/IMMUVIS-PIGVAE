import torch

from src.downstream.encode import _build_pl_module
from src.explainability.encoder_attribution import encoder_ig, pca_to_markers


def _small_encoder():
    # vae16_fb0p0_film: grid 16 -> 256 nodes, pca128 -> F=128, z_global width 512.
    return _build_pl_module("vae16_fb0p0_film").graph_ae.eval()


def test_encoder_ig_shape_and_finite():
    torch.manual_seed(0)
    gae = _small_encoder()
    node_features = torch.randn(2, 256, 128)
    attr = encoder_ig(gae, node_features, target=0, n_steps=8)
    assert attr.shape == (2, 256, 128)
    assert torch.isfinite(attr).all()
    # A non-degenerate attribution should not be identically zero.
    assert attr.abs().sum() > 0


def test_encoder_ig_completeness_delta_finite():
    torch.manual_seed(1)
    gae = _small_encoder()
    node_features = torch.randn(2, 256, 128)
    attr, delta = encoder_ig(
        gae, node_features, target=3, n_steps=16, return_convergence_delta=True
    )
    assert attr.shape == (2, 256, 128)
    assert torch.isfinite(delta).all()  # IG completeness gap is well-defined (finite)


def test_pca_to_markers_linear_map():
    # attribution in PCA space [B, N, F] -> marker space [B, N, C] via the loadings [F, C].
    torch.manual_seed(2)
    attr_pca = torch.randn(2, 256, 128)
    components = torch.randn(128, 768)  # stand-in for ipca.components_
    marker_attr = pca_to_markers(attr_pca, components)
    assert marker_attr.shape == (2, 256, 768)
    # exactness of the linear map: row 0, node 0 equals the explicit matmul
    expected = attr_pca[0, 0] @ components
    assert torch.allclose(marker_attr[0, 0], expected, atol=1e-5)
