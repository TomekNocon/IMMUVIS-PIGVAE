import torch
from torch.nn import MSELoss


class GraphReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_loss = MSELoss()  # BCEWithLogitsLoss() #MSE

    def forward(self, graph_true, graph_pred):
        # Use the mask to identify valid nodes
        device = graph_pred.node_features.device
        mask = graph_true.mask
        mask = mask.to(device)
        # Extract the node features for the true and predicted graphs, filtered
        # by the mask
        nodes_true = graph_true.out_node_features.to(device)
        nodes_true = nodes_true[mask]
        nodes_pred = graph_pred.node_features[mask]

        # Compute the node-based loss
        node_loss = self.node_loss(input=nodes_pred, target=nodes_true)

        # Return the loss dictionary
        loss = {"node_loss": node_loss, "loss": node_loss}
        return loss


# class PropertyLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse_loss = MSELoss()

#     def forward(self, input, target):
#         loss = self.mse_loss(input=input, target=target)
#         return loss


# class PermutationMatrixPenalty(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     @staticmethod
#     def entropy(p, axis, normalize=True, eps=10e-12):
#         if normalize:
#             p = p / (p.sum(axis=axis, keepdim=True) + eps)
#         e = -torch.sum(p * torch.clamp_min(torch.log(p), -100), axis=axis)
#         return e

#     def forward(self, perm, eps=10e-8):
#         # print(perm.shape)
#         perm = perm + eps
#         entropy_col = self.entropy(perm, axis=1, normalize=False)
#         entropy_row = self.entropy(perm, axis=2, normalize=False)
#         loss = entropy_col.mean() + entropy_row.mean()
#         return loss

class PermutationMatrixPenalty(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, perm):
        batch_size, num_nodes, _ = perm.shape
        
        # 1. Row sum constraint (each row should sum to 1)
        row_sum = perm.sum(dim=2)
        row_loss = torch.mean((row_sum - 1.0) ** 2)
        
        # 2. Column sum constraint (each column should sum to 1)
        col_sum = perm.sum(dim=1)
        col_loss = torch.mean((col_sum - 1.0) ** 2)
        
        # 3. Binary constraint (values should be close to 0 or 1)
        binary_loss = torch.mean(perm * (1 - perm))
        
        # 4. Entropy loss (encourage deterministic assignments)
        entropy_col = self.entropy(perm, axis=1, normalize=False)
        entropy_row = self.entropy(perm, axis=2, normalize=False)
        entropy_loss = (entropy_col.mean() + entropy_row.mean()) / 2
        
        # 5. Double stochastic constraint (optional)
        # This helps ensure the matrix is doubly stochastic
        double_stochastic_loss = torch.mean((perm @ perm.transpose(1, 2) - torch.eye(num_nodes, device=perm.device).unsqueeze(0)) ** 2)
        
        # Combine losses with appropriate weights
        total_loss = (
            row_loss + 
            col_loss + 
            0.1 * binary_loss +  # Lower weight for binary constraint
            0.1 * entropy_loss +  # Lower weight for entropy
            0.1 * double_stochastic_loss  # Lower weight for double stochastic
        )
        
        return total_loss

    @staticmethod
    def entropy(p, axis, normalize=True, eps=1e-12):
        if normalize:
            p = p / (p.sum(axis=axis, keepdim=True) + eps)
        e = -torch.sum(p * torch.clamp_min(torch.log(p + eps), -100), axis=axis)
        return e


class KLDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)
        loss = torch.mean(loss)
        return loss


class ContrastivePermutationLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.permutation_penalty = PermutationMatrixPenalty()
        
    def contrastive_loss(self, features, perm):
        # Normalize features
        features = torch.nn.functional.normalize(features, dim=-1)
        
        # Get positive pairs (features and their permuted versions)
        pos_features = torch.matmul(perm, features)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, pos_features.transpose(-2, -1)) / self.temperature
        
        # Get positive pairs (diagonal)
        pos_sim = torch.diagonal(sim_matrix, dim1=-2, dim2=-1)
        
        # Get negative pairs (off-diagonal)
        neg_sim = sim_matrix - torch.diag_embed(pos_sim)
        
        # Compute contrastive loss
        pos_loss = -pos_sim.mean()
        neg_loss = torch.logsumexp(neg_sim, dim=-1).mean()
        
        return pos_loss + neg_loss
        
    def forward(self, features, perm, epoch):
        # Get base permutation loss
        perm_loss = self.permutation_penalty(perm)
        
        # Get contrastive loss
        contrast_loss = self.contrastive_loss(features, perm)
        
        # Curriculum learning: gradually increase contrastive loss weight
        contrast_weight = min(0.1 * (epoch / 10), 0.1)
        
        # Combine losses
        total_loss = perm_loss + contrast_weight * contrast_loss
        
        return total_loss
