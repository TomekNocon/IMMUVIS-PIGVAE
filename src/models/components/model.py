import os
from typing import Any

import rootutils
import torch
from omegaconf import DictConfig

from src.data.components.graphs_datamodules import DenseGraphBatch
from src.models.components.losses import (
    CosineSimilarityLoss,
    GraphReconstructionLoss,
    KLDLoss,
    LaplacianLoss,
    MAELoss,
    MSEGraphLoss,
    MSEGridLoss,
    PermutationLoss,
    SignalToNoiseRatioLoss,
)

rootutils.setup_root(os.getcwd(), indicator=".project-root", pythonpath=True)


class Critic(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        # KL weight (scale) to balance reconstruction vs regularization
        self.kld_scale = float(getattr(hparams, "kld_loss_scale", 1.0))
        # self.beta = hparams.perm_loss_scale
        # self.gamma = hparams.contrastive_loss_scale
        self.vae = hparams.vae

        # Initialize reconstruction loss with Huber + Cosine + Gradient
        self.reconstruction_loss = GraphReconstructionLoss(
            # loss_alpha=HuberLoss(beta=hparams.huber_beta),
            loss_alpha=MSEGraphLoss(),
            loss_beta=CosineSimilarityLoss(),
            loss_gamma=LaplacianLoss(),
            alpha=hparams.alpha_scale,
            beta=hparams.beta_scale,
            gamma=hparams.gamma_scale,
        )

        self.kld_loss = KLDLoss(
            normalize_by_latent_dim=True, free_bits=hparams.get("kld_free_bits", 0.0)
        )

        self.permutation_loss = PermutationLoss()
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
        # KL warmup/anneal factor; if None, defaults to 1.0
        kld_alpha: float | None = None,
        soft_probs: torch.Tensor | None = None,
        perm: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        recon_loss = self.reconstruction_loss(graph_true=graph_true, graph_pred=graph_pred)
        # contrastive_loss = self.contrastive_loss(graph_emb)
        permutation_loss = self.permutation_loss(
            soft_probs if soft_probs is not None else None  # perm
        )

        mae_loss = self.mae_loss(graph_true=graph_true, graph_pred=graph_pred)
        mse_loss = self.mse_loss(graph_true=graph_true, graph_pred=graph_pred)
        signal_to_noise_ratio_loss = self.signal_to_noise_ratio_loss(
            graph_true=graph_true, graph_pred=graph_pred
        )

        loss = {
            **recon_loss,
            # "contrastive_loss": contrastive_loss,
            "permutation_loss": permutation_loss,
            "mae_loss": mae_loss,
            "signal_to_noise_ratio_loss": signal_to_noise_ratio_loss,
            "mse_loss": mse_loss,
        }
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
        # KL warmup/anneal factor; if None, defaults to 1.0
        kld_alpha: float | None = None,
        prefix: str | None = None,
        soft_probs: torch.Tensor | None = None,
        perm: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        loss = self(
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
        metrics = loss

        if prefix is not None:
            metrics2 = {}
            for key in metrics.keys():
                new_key = prefix + "_" + str(key)
                metrics2[new_key] = metrics[key]
            metrics = metrics2
        return metrics


if __name__ == "__main__":
    pass
