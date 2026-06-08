import torch
import torch.nn as nn

from src.utils.inspection.reconstruction import image_space_reconstruction


class _StubPCA(nn.Module):
    """Minimal PCALayer-like inverse: z[B,N,k] -> z @ components + pca_mean ([B,N,D])."""

    def __init__(self, k: int = 4, d: int = 16):
        super().__init__()
        self.register_buffer("components", torch.randn(k, d))
        self.register_buffer("pca_mean", torch.randn(d))

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return torch.matmul(z, self.components) + self.pca_mean


def test_image_space_zero_for_identical():
    pca = _StubPCA(k=4, d=16)
    z = torch.randn(2, 5, 4)
    out = image_space_reconstruction(z.clone(), z, pca)
    assert out["n_orig_channels"] == 16
    assert out["image_mse"] < 1e-10
    assert out["image_mae"] < 1e-6
    assert out["image_r2_mean"] > 0.999


def test_image_space_error_is_positive_and_projected():
    pca = _StubPCA(k=4, d=16)
    target = torch.randn(3, 6, 4)
    pred = target + 0.1 * torch.randn_like(target)
    out = image_space_reconstruction(pred, target, pca)
    assert out["image_mse"] > 0
    assert out["n_orig_channels"] == 16
    assert "image_r2_mean" in out and "image_r2_min" in out
