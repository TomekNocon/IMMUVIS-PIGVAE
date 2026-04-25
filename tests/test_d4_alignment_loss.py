import math
import torch
import pytest

from src.data.components.graphs_datamodules import DenseGraphBatch
from src.models.components.losses import D4AlignmentLoss, PerSampleReconLoss


def make_batch(B: int = 4, grid_size: int = 6, C: int = 8):
    N = grid_size * grid_size
    return DenseGraphBatch(node_features=torch.randn(B, N, C), edge_features=torch.empty(0))


def make_d4_loss(grid_size=6, alpha=1.0, beta=0.0, gamma=0.0):
    recon = PerSampleReconLoss(grid_size=grid_size, huber_beta=1.0, alpha=alpha, beta=beta, gamma=gamma)
    return D4AlignmentLoss(grid_size=grid_size, reconstruction_loss=recon)


class TestPerSampleReconLoss:
    def test_output_shape(self):
        loss_fn = PerSampleReconLoss(grid_size=6, huber_beta=1.0, alpha=1.0, beta=0.1, gamma=0.001)
        pred = torch.randn(4, 36, 8)
        target = torch.randn(4, 36, 8)
        out = loss_fn(pred, target)
        assert out.shape == (4,), f"expected [B], got {out.shape}"

    def test_zero_loss_for_identical_inputs(self):
        loss_fn = PerSampleReconLoss(grid_size=6, huber_beta=1.0, alpha=1.0, beta=0.0, gamma=0.0)
        x = torch.randn(4, 36, 8)
        out = loss_fn(x, x)
        assert out.abs().max().item() < 1e-5

    def test_gradients_flow(self):
        loss_fn = PerSampleReconLoss(grid_size=6, huber_beta=1.0, alpha=1.0, beta=0.1, gamma=0.001)
        pred = torch.randn(4, 36, 8, requires_grad=True)
        target = torch.randn(4, 36, 8)
        loss_fn(pred, target).sum().backward()
        assert pred.grad is not None


class TestD4AlignmentLoss:
    def test_output_is_dict_with_loss(self):
        loss_fn = make_d4_loss()
        out = loss_fn(make_batch(), make_batch())
        assert "loss" in out and "d4_alignment_loss" in out
        assert out["loss"].ndim == 0

    def test_loss_is_positive(self):
        out = make_d4_loss()(make_batch(), make_batch())
        assert out["loss"].item() >= 0

    def test_perfect_reconstruction_zero(self):
        """Constant (D4-invariant) input: pred==gt gives zero loss for all 8 orientations."""
        loss_fn = make_d4_loss(alpha=1.0, beta=0.0, gamma=0.0)
        gt_features = torch.ones(2, 36, 8)
        gt = DenseGraphBatch(node_features=gt_features, edge_features=torch.empty(0))
        pred = DenseGraphBatch(node_features=gt_features.clone(), edge_features=torch.empty(0))
        assert abs(loss_fn(gt, pred)["loss"].item()) < 1e-4

    def test_perm_matrices_shape(self):
        assert make_d4_loss().perm_matrices.shape == (8, 36, 36)

    def test_perm_matrices_are_orthogonal(self):
        loss_fn = make_d4_loss()
        I = torch.eye(36)
        for k in range(8):
            P = loss_fn.perm_matrices[k]
            assert torch.allclose(P @ P.T, I, atol=1e-5), f"P[{k}] not orthogonal"

    def test_d4_loss_has_no_loss_params(self):
        """D4AlignmentLoss should not hold huber_beta / alpha / beta / gamma directly."""
        loss_fn = make_d4_loss()
        assert not hasattr(loss_fn, "huber_beta")
        assert not hasattr(loss_fn, "alpha")

    def test_gradients_flow(self):
        loss_fn = make_d4_loss()
        gt = make_batch()
        pred_features = torch.randn(4, 36, 8, requires_grad=True)
        pred = DenseGraphBatch(node_features=pred_features, edge_features=torch.empty(0))
        loss_fn(gt, pred)["loss"].backward()
        assert pred_features.grad is not None
        assert not torch.isnan(pred_features.grad).any()
