"""The PIGVAE training objective (reconstruction + KL + permutation diversity) and its metrics."""

from typing import Any

import torch
from omegaconf import DictConfig

from src.data.components.graphs_datamodules import DenseGraphBatch
from src.models.components.losses import (
    CosineSimilarityLoss,
    GradientLoss,
    GraphReconstructionLoss,
    HuberLoss,
    KLDLoss,
    MAELoss,
    MSEGridLoss,
    PermutationLoss,
    SignalToNoiseRatioLoss,
)


class Critic(torch.nn.Module):
    """Computes the training loss and the logged reconstruction metrics.

    ``loss = alpha * Huber + beta * (1 - cosine) + gamma * gradient-L1
    + perm_weight * PermutationLoss + kld_scale * kld_alpha * KL`` (KL only in VAE mode).
    MAE, MSE and SNR are returned for logging only.
    """

    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.kld_scale = float(getattr(hparams, "kld_loss_scale", 1.0))
        self.vae = hparams.vae
        self.reconstruction_loss = GraphReconstructionLoss(
            loss_alpha=HuberLoss(beta=hparams.huber_beta),
            loss_beta=CosineSimilarityLoss(),
            loss_gamma=GradientLoss(),
            alpha=hparams.alpha_scale,
            beta=hparams.beta_scale,
            gamma=hparams.gamma_scale,
        )
        self.kld_loss = KLDLoss(
            normalize_by_latent_dim=True, free_bits=hparams.get("kld_free_bits", 0.0)
        )
        self.permutation_loss = PermutationLoss(
            num_permutations=int(getattr(hparams, "num_aug_per_sample", 8))
        )
        self.mae_loss = MAELoss()
        self.signal_to_noise_ratio_loss = SignalToNoiseRatioLoss()
        self.mse_loss = MSEGridLoss()

    def forward(
        self,
        graph_emb: torch.Tensor,
        graph_true: DenseGraphBatch,
        graph_pred: DenseGraphBatch,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 0.0,
        kld_alpha: float | None = None,
        soft_probs: torch.Tensor | None = None,
        perm: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Loss terms and metrics; ``loss["loss"]`` is the differentiable total.

        Args:
            beta: Weight of the permutation-diversity loss.
            kld_alpha: KL annealing factor (``None`` = 1).
        """
        loss = self.reconstruction_loss(graph_true=graph_true, graph_pred=graph_pred)
        permutation_loss = self.permutation_loss(soft_probs)
        if not isinstance(permutation_loss, torch.Tensor) or not permutation_loss.is_cuda:
            permutation_loss = permutation_loss.to(graph_pred.node_features.device)

        loss.update({
            "permutation_loss": permutation_loss,
            "mae_loss": self.mae_loss(graph_true=graph_true, graph_pred=graph_pred),
            "signal_to_noise_ratio_loss": self.signal_to_noise_ratio_loss(
                graph_true=graph_true, graph_pred=graph_pred
            ),
            "mse_loss": self.mse_loss(graph_true=graph_true, graph_pred=graph_pred),
        })
        loss["loss"] = loss["loss"] + beta * permutation_loss
        if self.vae:
            kld_loss = self.kld_loss(mu, logvar)
            loss["kld_loss"] = kld_loss
            scale = self.kld_scale * (1.0 if kld_alpha is None else float(kld_alpha))
            loss["loss"] = loss["loss"] + scale * kld_loss
        return loss

    def evaluate(
        self,
        graph_emb: torch.Tensor,
        graph_true: DenseGraphBatch,
        graph_pred: DenseGraphBatch,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 0.0,
        kld_alpha: float | None = None,
        prefix: str | None = None,
        soft_probs: torch.Tensor | None = None,
        perm: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """:meth:`forward` with every key prefixed by ``f"{prefix}_"``."""
        metrics = self(
            graph_emb=graph_emb,
            graph_true=graph_true,
            graph_pred=graph_pred,
            soft_probs=soft_probs,
            perm=perm,
            beta=beta,
            kld_alpha=kld_alpha,
            mu=mu,
            logvar=logvar,
        )
        if prefix is None:
            return metrics
        return {f"{prefix}_{key}": value for key, value in metrics.items()}
