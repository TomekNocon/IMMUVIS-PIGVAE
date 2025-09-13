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


# class ContrastiveLoss(torch.nn.Module):
#     def __init__(self, temperature: float = 0.07):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, features: torch.Tensor) -> torch.Tensor:
#         # computing contrastive loss as in SimCLR
#         batch_size = features.shape[0] // 2  # batch_size * num_markers

#         labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
#         labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
#         labels = labels.to(features.device)
#         features = F.normalize(features, dim=-1)
#         similarity_matrix = torch.matmul(features, features.T)

#         # discard the main diagonal from both: labels and similarity matrix
#         mask = torch.eye(labels.shape[0], dtype=torch.bool).to(labels.device)
#         labels = labels[~mask].view(labels.shape[0], -1)

#         similarity_matrix = similarity_matrix[~mask].view(
#             similarity_matrix.shape[0], -1
#         )
#         # l_{i,j} = -sim(z_i,z_j)/t + log[{∑^{N}_{k=1}1_{[k \neq i]} exp(sim(z_i,z_k)/t)}]$

#         positives = similarity_matrix[labels.bool()].view(batch_size * 2, -1)
#         negatives = similarity_matrix[~labels.bool()].view(batch_size * 2, -1)
#         positives = -positives / self.temperature
#         negatives = torch.logsumexp(negatives / self.temperature, dim=1)
#         loss = positives + negatives
#         loss = loss.mean()

#         return loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.07, num_aug_per_sample: int = 8):
        super().__init__()
        self.temperature = temperature
        self.num_aug_per_sample = num_aug_per_sample - 1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # computing contrastive loss as in SimCLR
        features = F.normalize(features, dim=-1)
        N = features.shape[0]
        samples_per_group = 1 + self.num_aug_per_sample
        original_batch_size = N // samples_per_group  # number of original images
        labels = torch.cat(
            [torch.arange(original_batch_size) for _ in range(samples_per_group)], dim=0
        )
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).bool()
        label_matrix = label_matrix.to(features.device)
        # # Step 4: Similarity matrix
        sim = torch.matmul(features, features.T)  # [N, N]
        sim = sim / self.temperature

        # Mask out self-similarity
        mask = torch.eye(N, dtype=torch.bool, device=features.device)
        sim.masked_fill_(mask, float("-inf"))  # ignore diagonal

        # Extract positives and negatives
        pos_mask = label_matrix & ~mask  # positives without self
        neg_mask = ~label_matrix  # everything else

        pos_sim = sim.masked_fill(~pos_mask, float("-inf"))
        neg_sim = sim.masked_fill(~neg_mask, float("-inf"))

        # Compute loss
        pos_term = torch.logsumexp(pos_sim, dim=1)  # [N]
        neg_term = torch.logsumexp(neg_sim, dim=1)  # [N]
        loss = -pos_term + neg_term

        return loss.mean()


class PermutationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs: torch.Tensor):
        if probs is None:
            return 0
        # logits: (batch_size, num_classes)
        avg_probs = probs.mean(dim=0)  # (num_classes,)
        log_avg_probs = torch.log(avg_probs + 1e-12)
        entropy = -torch.sum(avg_probs * log_avg_probs)  # scalar
        max_entropy = torch.log(torch.tensor(avg_probs.size(0), dtype=avg_probs.dtype, device=avg_probs.device))
        return max_entropy - entropy  # always positive, minimizing this maximizes entropy


class PermutaionMatrixLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def entropy(
        p: torch.Tensor, axis: int, normalize: bool = True, eps=10e-12
    ) -> torch.Tensor:
        if normalize:
            p = p / (p.sum(axis=axis, keepdim=True) + eps)
        e = -torch.sum(p * torch.clamp_min(torch.log(p), -100), axis=axis)
        return e

    def forward(self, perm: torch.Tensor, eps: float = 10e-8) -> torch.Tensor:
        if not perm:
            return 0
        perm = perm + eps
        entropy_col = self.entropy(perm, axis=1, normalize=False)
        entropy_row = self.entropy(perm, axis=2, normalize=False)
        loss = entropy_col.mean() + entropy_row.mean()
        return loss
