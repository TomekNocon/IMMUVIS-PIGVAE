import torch
from src.models.components.losses import (
    GraphReconstructionLoss,
    KLDLoss,
    ContrastiveLoss,
    PermutationLoss,
)
from omegaconf import DictConfig
from src.data.components.graphs_datamodules import DenseGraphBatch
from typing import Dict, Any, Optional

import os
import rootutils

rootutils.setup_root(os.getcwd(), indicator=".project-root", pythonpath=True)


class Critic(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.alpha = hparams.kld_loss_scale
        # self.beta = hparams.perm_loss_scale
        self.gamma = hparams.contrastive_loss_scale
        self.vae = hparams.vae
        self.reconstruction_loss = GraphReconstructionLoss()
        self.contrastive_loss = ContrastiveLoss(temperature=hparams.temperature)
        self.kld_loss = KLDLoss()
        self.permutation_loss = PermutationLoss()

    def forward(
        self,
        graph_emb: torch.Tensor,
        graph_true: DenseGraphBatch,
        graph_pred: DenseGraphBatch,
        soft_probs: torch.Tensor,
        beta: float,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Dict[str, Any]:
        recon_loss = self.reconstruction_loss(
            graph_true=graph_true, graph_pred=graph_pred
        )
        contrastive_loss = self.contrastive_loss(graph_emb)
        permutation_loss = self.permutation_loss(soft_probs)
        loss = {
            **recon_loss,
            "contrastive_loss": contrastive_loss,
            "permutation_loss": permutation_loss,
        }
        loss["loss"] = (
            loss["loss"] + beta * permutation_loss + self.gamma * contrastive_loss
        )
        if self.vae:
            kld_loss = self.kld_loss(mu, logvar)
            loss["kld_loss"] = kld_loss
            loss["loss"] = loss["loss"] + self.alpha * kld_loss
        return loss

    def evaluate(
        self,
        graph_emb: torch.Tensor,
        graph_true: DenseGraphBatch,
        graph_pred: DenseGraphBatch,
        soft_probs: torch.Tensor,
        beta: float,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        loss = self(
            graph_emb=graph_emb,
            graph_true=graph_true,
            graph_pred=graph_pred,
            soft_probs=soft_probs,
            beta=beta,
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
    _ = Critic()
