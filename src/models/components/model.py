import torch
from src.models.components.losses import (
    GraphReconstructionLoss,
    PermutaionMatrixPenalty,
    PropertyLoss,
    KLDLoss
)
from omegaconf import DictConfig

import os
import rootutils
rootutils.setup_root(os.getcwd(), indicator=".project-root", pythonpath=True)


class Critic(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.alpha = hparams.kld_loss_scale
        self.beta = hparams.perm_loss_scale
        self.gamma = hparams.property_loss_scale
        self.vae = hparams.vae
        self.reconstruction_loss = GraphReconstructionLoss()
        self.perm_loss = PermutaionMatrixPenalty()
        self.property_loss = PropertyLoss()
        self.kld_loss = KLDLoss()

    def forward(self, graph_true, graph_pred, perm, mu, logvar):
        recon_loss = self.reconstruction_loss(
            graph_true=graph_true, graph_pred=graph_pred
        )
        perm_loss = self.perm_loss(perm)
        property_loss = self.property_loss(
            input=graph_pred.properties, target=graph_true.properties
        )
        loss = {**recon_loss, "perm_loss": perm_loss, "property_loss": property_loss}
        loss["loss"] = loss["loss"] + self.beta * perm_loss + self.gamma * property_loss
        if self.vae:
            kld_loss = self.kld_loss(mu, logvar)
            loss["kld_loss"] = kld_loss
            loss["loss"] = loss["loss"] + self.alpha * kld_loss
        return loss

    def evaluate(self, graph_true, graph_pred, perm, mu, logvar, prefix=None):
        loss = self(
            graph_true=graph_true,
            graph_pred=graph_pred,
            perm=perm,
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