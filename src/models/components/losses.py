import torch
from torch.nn import MSELoss
import torch.nn.functional as F
from src.data.components.graphs_datamodules import DenseGraphBatch
from typing import Dict, Any


class GraphReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_loss = MSELoss()  # BCEWithLogitsLoss() #MSE

    def forward(
        self, graph_true: DenseGraphBatch, graph_pred: DenseGraphBatch
    ) -> Dict[str, Any]:
        # Use the mask to identify valid nodes
        device = graph_pred.node_features.device
        mask = graph_true.mask
        mask = mask.to(device)
        # Extract the node features for the true and predicted graphs, filtered
        # by the mask
        nodes_true = graph_true.node_features.to(device)
        nodes_true = nodes_true[mask]
        nodes_pred = graph_pred.node_features[mask]

        # Compute the node-based loss
        node_loss = self.node_loss(input=nodes_pred, target=nodes_true)

        # Return the loss dictionary
        loss = {"node_loss": node_loss, "loss": node_loss}
        return loss


class KLDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)
        loss = torch.mean(loss)
        return loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # computing contrastive loss as in SimCLR
        batch_size = features.shape[0] // 2  # batch_size * num_markers

        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)
        features = F.normalize(features, dim=-1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarity matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(labels.device)
        labels = labels[~mask].view(labels.shape[0], -1)

        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )
        # l_{i,j} = -sim(z_i,z_j)/t + log[{∑^{N}_{k=1}1_{[k \neq i]} exp(sim(z_i,z_k)/t)}]$

        positives = similarity_matrix[labels.bool()].view(batch_size * 2, -1)
        negatives = similarity_matrix[~labels.bool()].view(batch_size * 2, -1)
        positives = -positives / self.temperature
        negatives = torch.logsumexp(negatives / self.temperature, dim=1)
        loss = positives + negatives
        loss = loss.mean()

        return loss


class PermutationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs: torch.Tensor):
        # logits: (batch_size, num_classes)
        avg_probs = probs.mean(dim=0)  # (num_classes,)
        log_avg_probs = torch.log(avg_probs + 1e-12)
        entropy = -torch.sum(avg_probs * log_avg_probs)  # scalar
        return -entropy  # negative entropy to maximize it
