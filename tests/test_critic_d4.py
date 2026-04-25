# tests/test_critic_d4.py
import torch
import pytest
from omegaconf import OmegaConf

from src.data.components.graphs_datamodules import DenseGraphBatch


def make_critic_hparams():
    return OmegaConf.create({
        "kld_loss_scale": 0.05,
        "kld_free_bits": 0.0,
        "vae": True,
        "grid_size": 4,
        "huber_beta": 2.0,
        "alpha_scale": 1.0,
        "beta_scale": 0.1,
        "gamma_scale": 0.001,
    })


def make_batch(B=2, grid_size=4, C=8):
    N = grid_size * grid_size
    return DenseGraphBatch(
        node_features=torch.randn(B, N, C),
        edge_features=torch.empty(0),
        mask=torch.ones(B, N, dtype=torch.bool),
    )


class TestCriticD4:
    def test_forward_returns_expected_keys(self):
        from src.models.components.model import Critic
        critic = Critic(make_critic_hparams())
        gt = make_batch()
        pred = make_batch()
        mu = torch.randn(2, 32)
        logvar = torch.zeros(2, 32)
        out = critic(
            graph_emb=torch.randn(2, 32),
            graph_true=gt,
            graph_pred=pred,
            mu=mu,
            logvar=logvar,
        )
        assert "loss" in out
        assert "d4_alignment_loss" in out
        assert "kld_loss" in out
        assert "mae_loss" in out
        assert "mse_loss" in out
        assert "signal_to_noise_ratio_loss" in out

    def test_no_permutation_loss_key(self):
        from src.models.components.model import Critic
        critic = Critic(make_critic_hparams())
        gt = make_batch()
        pred = make_batch()
        out = critic(
            graph_emb=torch.randn(2, 32),
            graph_true=gt,
            graph_pred=pred,
            mu=torch.randn(2, 32),
            logvar=torch.zeros(2, 32),
        )
        assert "permutation_loss" not in out

    def test_evaluate_adds_prefix(self):
        from src.models.components.model import Critic
        critic = Critic(make_critic_hparams())
        gt = make_batch()
        pred = make_batch()
        out = critic.evaluate(
            graph_emb=torch.randn(2, 32),
            graph_true=gt,
            graph_pred=pred,
            mu=torch.randn(2, 32),
            logvar=torch.zeros(2, 32),
            prefix="val",
        )
        assert "val_loss" in out
        assert "val_d4_alignment_loss" in out

    def test_d4_loss_is_injected_per_sample_loss(self):
        """Critic must delegate to PerSampleReconLoss, not re-implement loss logic."""
        from src.models.components.model import Critic
        from src.models.components.losses import PerSampleReconLoss
        critic = Critic(make_critic_hparams())
        assert isinstance(critic.d4_alignment_loss.reconstruction_loss, PerSampleReconLoss)
