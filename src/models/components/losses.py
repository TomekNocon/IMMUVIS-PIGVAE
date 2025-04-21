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
        nodes_true = graph_true.node_features.to(device)
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


class PermutationMatrixPenalty(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def entropy(p, axis, normalize=True, eps=10e-12):
        if normalize:
            p = p / (p.sum(axis=axis, keepdim=True) + eps)
        e = -torch.sum(p * torch.clamp_min(torch.log(p), -100), axis=axis)
        return e

    def forward(self, perm, eps=10e-8):
        # print(perm.shape)
        perm = perm + eps
        entropy_col = self.entropy(perm, axis=1, normalize=False)
        entropy_row = self.entropy(perm, axis=2, normalize=False)
        loss = entropy_col.mean() + entropy_row.mean()
        return loss


class KLDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)
        loss = torch.mean(loss)
        return loss
