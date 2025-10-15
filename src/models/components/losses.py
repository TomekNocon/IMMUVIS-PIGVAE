import torch
from torch.nn import MSELoss, L1Loss, CosineSimilarity
import torch.nn.functional as F
from src.data.components.graphs_datamodules import DenseGraphBatch
from typing import Dict, Any


class GraphReconstructionLoss(torch.nn.Module):
    def __init__(self, use_gradient_loss: bool = False, gradient_loss_weight: float = 0.1):
        """
        Reconstruction loss with optional gradient/detail preservation.
        
        Args:
            use_gradient_loss: If True, adds gradient-based loss to preserve details
            gradient_loss_weight: Weight for gradient loss term
        """
        super().__init__()
        self.node_loss = MSELoss()  # BCEWithLogitsLoss() #MSE
        self.use_gradient_loss = use_gradient_loss
        self.gradient_loss_weight = gradient_loss_weight

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

        total_loss = node_loss
        
        # Add gradient loss to preserve high-frequency details
        if self.use_gradient_loss:
            # Reshape to grid for gradient computation
            # Assuming nodes are in grid order: [batch*aug, num_nodes, features]
            B = graph_true.node_features.shape[0]
            N = graph_true.node_features.shape[1]
            grid_size = int(N ** 0.5)
            
            if grid_size * grid_size == N:  # Verify it's a square grid
                pred_grid = graph_pred.node_features.view(B, grid_size, grid_size, -1)
                true_grid = graph_true.node_features.view(B, grid_size, grid_size, -1)
                
                # Compute gradients in both directions
                pred_grad_x = pred_grid[:, :, 1:, :] - pred_grid[:, :, :-1, :]
                pred_grad_y = pred_grid[:, 1:, :, :] - pred_grid[:, :-1, :, :]
                
                true_grad_x = true_grid[:, :, 1:, :] - true_grid[:, :, :-1, :]
                true_grad_y = true_grid[:, 1:, :, :] - true_grid[:, :-1, :, :]
                
                # L1 loss on gradients (preserves sharp edges better than L2)
                gradient_loss = (
                    torch.mean(torch.abs(pred_grad_x - true_grad_x)) +
                    torch.mean(torch.abs(pred_grad_y - true_grad_y))
                )
                
                total_loss = node_loss + self.gradient_loss_weight * gradient_loss
                
                # Return the loss dictionary
                loss = {
                    "node_loss": node_loss, 
                    "gradient_loss": gradient_loss,
                    "loss": total_loss
                }
                return loss

        # Return the loss dictionary (no gradient loss)
        loss = {"node_loss": node_loss, "loss": total_loss}
        return loss


class MAELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_loss = L1Loss()  # BCEWithLogitsLoss() #MSE

    def forward(
        self, graph_true: DenseGraphBatch, graph_pred: DenseGraphBatch
    ) -> torch.Tensor:
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
        loss = self.node_loss(input=nodes_pred, target=nodes_true)

        return loss


class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self, return_as_loss: bool = True):
        """
        Args:
            return_as_loss: If True, returns (1 - cosine_similarity) so lower is better.
                          If False, returns cosine_similarity directly (higher is better).
        """
        super().__init__()
        self.node_loss = CosineSimilarity()
        self.return_as_loss = return_as_loss

    def forward(
        self, graph_true: DenseGraphBatch, graph_pred: DenseGraphBatch
    ) -> torch.Tensor:
        # Use the mask to identify valid nodes
        device = graph_pred.node_features.device
        mask = graph_true.mask
        mask = mask.to(device)
        # Extract the node features for the true and predicted graphs, filtered
        # by the mask
        nodes_true = graph_true.node_features.to(device)
        nodes_true = nodes_true[mask]
        nodes_pred = graph_pred.node_features[mask]

        # Compute cosine similarity (range: -1 to 1, where 1 = identical)
        similarity = self.node_loss(nodes_pred.flatten(1), nodes_true.flatten(1)).mean()
        
        if self.return_as_loss:
            # Convert to loss: 1 - similarity, so 0 = perfect, 2 = worst
            return 1 - similarity
        else:
            # Return as metric: higher = better
            return similarity


class SignalToNoiseRatioLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        graph_true: DenseGraphBatch,
        graph_pred: DenseGraphBatch,
        eps: float = 1e-8,
    ) -> torch.Tensor:
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
        signal_power = torch.mean(nodes_true**2)
        noise_power = torch.mean((nodes_pred - nodes_true) ** 2)

        loss = 10 * torch.log10(signal_power / (noise_power + eps))

        return loss


class KLDLoss(torch.nn.Module):
    def __init__(self, normalize_by_latent_dim: bool = True, free_bits: float = 0.0):
        """
        KLD Loss with optional free bits to prevent posterior collapse.
        
        Args:
            normalize_by_latent_dim: If True, average over latent dims instead of sum
            free_bits: Free bits threshold - KLD below this per dimension is not penalized.
                      Typical values: 0.0 (disabled), 0.5, 1.0, 2.0
        """
        super().__init__()
        self.normalize_by_latent_dim = normalize_by_latent_dim
        self.free_bits = free_bits

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        if self.free_bits > 0:
            # Apply free bits: max(KLD_per_dim, free_bits)
            # This prevents over-compression of the latent space
            kld_per_dim = torch.clamp(kld_per_dim, min=self.free_bits)
        
        if self.normalize_by_latent_dim:
            # Average over latent dimensions, then average over batch
            loss = torch.mean(kld_per_dim)
        else:
            # Sum over latent dimensions, then average over batch (original behavior)
            loss = torch.sum(kld_per_dim, dim=1).mean()
        
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
        max_entropy = torch.log(
            torch.tensor(
                avg_probs.size(0), dtype=avg_probs.dtype, device=avg_probs.device
            )
        )
        return (
            max_entropy - entropy
        )  # always positive, minimizing this maximizes entropy


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
