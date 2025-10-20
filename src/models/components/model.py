import torch
from src.models.components.losses import (
    GraphReconstructionLoss,
    KLDLoss,
    PermutationLoss,
    MAELoss,
    CosineSimilarityLoss,
    SignalToNoiseRatioLoss,
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
        # self.gamma = hparams.contrastive_loss_scale
        self.vae = hparams.vae
        
        # Initialize reconstruction loss with gradient preservation and cosine similarity
        self.reconstruction_loss = GraphReconstructionLoss(
            use_gradient_loss=hparams.get('use_gradient_loss', True),
            gradient_loss_weight=hparams.get('gradient_loss_weight', 0.5),
            use_cosine_loss=hparams.get('use_cosine_loss', True),
            cosine_loss_weight=hparams.get('cosine_loss_weight', 0.1)
        )
        
        # self.contrastive_loss = ContrastiveLoss(
        #     temperature=hparams.temperature,
        #     num_aug_per_sample=hparams.num_aug_per_sample,
        # )
        
        # Initialize KLD loss with free bits to prevent posterior collapse
        self.kld_loss = KLDLoss(
            normalize_by_latent_dim=True,
            free_bits=hparams.get('kld_free_bits', 0.0)
        )
        
        self.permutation_loss = PermutationLoss()
        self.mae_loss = MAELoss()
        self.cosine_similarity_loss = CosineSimilarityLoss()
        self.signal_to_noise_ratio_loss = SignalToNoiseRatioLoss()

    def forward(
        self,
        graph_emb: torch.Tensor,
        graph_true: DenseGraphBatch,
        graph_pred: DenseGraphBatch,
        beta: float,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        soft_probs: Optional[torch.Tensor] = None,
        perm: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        recon_loss = self.reconstruction_loss(
            graph_true=graph_true, graph_pred=graph_pred
        )
        # contrastive_loss = self.contrastive_loss(graph_emb)
        permutation_loss = self.permutation_loss(
            soft_probs if soft_probs is not None else perm
        )

        mae_loss = self.mae_loss(graph_true=graph_true, graph_pred=graph_pred)
        cosine_similarity_loss = self.cosine_similarity_loss(
            graph_true=graph_true, graph_pred=graph_pred
        )
        signal_to_noise_ratio_loss = self.signal_to_noise_ratio_loss(
            graph_true=graph_true, graph_pred=graph_pred
        )

        loss = {
            **recon_loss,
            # "contrastive_loss": contrastive_loss,
            "permutation_loss": permutation_loss,
            "mae_loss": mae_loss,
            "cosine_similarity_loss": cosine_similarity_loss,
            "signal_to_noise_ratio_loss": signal_to_noise_ratio_loss,
        }
        loss["loss"] = (
            loss["loss"] + beta * permutation_loss  # + self.gamma * contrastive_loss
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
        beta: float,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        prefix: Optional[str] = None,
        soft_probs: Optional[torch.Tensor] = None,
        perm: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        loss = self(
            graph_emb=graph_emb,
            graph_true=graph_true,
            graph_pred=graph_pred,
            soft_probs=soft_probs,
            perm=perm,
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
