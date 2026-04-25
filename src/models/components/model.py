import os
from typing import Any

import rootutils
import torch
from omegaconf import DictConfig

from src.data.components.graphs_datamodules import DenseGraphBatch
from src.models.components.losses import (
    D4AlignmentLoss,
    KLDLoss,
    MAELoss,
    MSEGridLoss,
    PerSampleReconLoss,
    SignalToNoiseRatioLoss,
)

rootutils.setup_root(os.getcwd(), indicator=".project-root", pythonpath=True)


class Critic(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.kld_scale = float(getattr(hparams, "kld_loss_scale", 1.0))
        self.vae = hparams.vae
        per_sample_loss = PerSampleReconLoss(
            grid_size=hparams.grid_size,
            huber_beta=hparams.huber_beta,
            alpha=hparams.alpha_scale,
            beta=hparams.beta_scale,
            gamma=hparams.gamma_scale,
        )
        self.d4_alignment_loss = D4AlignmentLoss(
            grid_size=hparams.grid_size,
            reconstruction_loss=per_sample_loss,
        )
        self.kld_loss = KLDLoss(
            normalize_by_latent_dim=True, free_bits=hparams.get("kld_free_bits", 0.0)
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
        kld_alpha: float | None = None,
    ) -> dict[str, Any]:
        loss = {
            **self.d4_alignment_loss(graph_true=graph_true, graph_pred=graph_pred),
            "mae_loss": self.mae_loss(graph_true=graph_true, graph_pred=graph_pred),
            "mse_loss": self.mse_loss(graph_true=graph_true, graph_pred=graph_pred),
            "signal_to_noise_ratio_loss": self.signal_to_noise_ratio_loss(
                graph_true=graph_true, graph_pred=graph_pred
            ),
        }
        if self.vae:
            kld = self.kld_loss(mu, logvar)
            loss["kld_loss"] = kld
            scale = self.kld_scale * (1.0 if kld_alpha is None else float(kld_alpha))
            loss["loss"] = loss["loss"] + scale * kld
        return loss

    def evaluate(
        self,
        graph_emb: torch.Tensor,
        graph_true: DenseGraphBatch,
        graph_pred: DenseGraphBatch,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kld_alpha: float | None = None,
        prefix: str | None = None,
    ) -> dict[str, Any]:
        loss = self(
            graph_emb=graph_emb,
            graph_true=graph_true,
            graph_pred=graph_pred,
            kld_alpha=kld_alpha,
            mu=mu,
            logvar=logvar,
        )
        if prefix is not None:
            loss = {prefix + "_" + k: v for k, v in loss.items()}
        return loss


if __name__ == "__main__":
    pass
