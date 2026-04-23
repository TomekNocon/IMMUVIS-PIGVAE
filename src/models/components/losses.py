import math
from typing import Any

import lpips
import pytorch_msssim
import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity, L1Loss, MSELoss

from src.data.components.graphs_datamodules import DenseGraphBatch


class BaseGridReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BaseReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GraphReconstructionLoss(torch.nn.Module):
    def __init__(
        self,
        loss_alpha: BaseReconstructionLoss | BaseGridReconstructionLoss,
        loss_beta: BaseReconstructionLoss | BaseGridReconstructionLoss,
        loss_gamma: BaseReconstructionLoss | BaseGridReconstructionLoss,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ):
        """
        components: List[Tuple[str, torch.nn.Module, float, bool]]
          Each item is (name, module, weight, use_grid)
        """
        super().__init__()
        self.losses = torch.nn.ModuleDict({
            "alpha_recon": loss_alpha,
            "beta_recon": loss_beta,
            "gamma_recon": loss_gamma,
        })
        self.weights = {
            "alpha_recon": alpha,
            "beta_recon": beta,
            "gamma_recon": gamma,
        }

    def forward(self, graph_true: DenseGraphBatch, graph_pred: DenseGraphBatch) -> dict[str, Any]:
        out = {}
        device = graph_pred.node_features.device
        true = graph_true.node_features.to(device)
        pred = graph_pred.node_features.to(device)

        b, n, d = graph_true.node_features.shape
        grid_h = math.isqrt(int(n))

        true_grid = true.view(b, grid_h, grid_h, d).permute(0, 3, 1, 2)
        pred_grid = pred.view(b, grid_h, grid_h, d).permute(0, 3, 1, 2)

        total = 0.0

        for name, loss_module in self.losses.items():
            if isinstance(loss_module, BaseReconstructionLoss):
                loss_val = loss_module(pred, true)
            else:
                loss_val = loss_module(pred_grid, true_grid)

            weighted = self.weights[name] * loss_val
            out[name] = weighted.item()
            total += weighted

        out["loss"] = total
        return out


# class GraphReconstructionLoss(torch.nn.Module):
#     def __init__(
#         self,
#         huber_beta: float = 1.0,
#         cosine_loss_weight: float = 0.1,
#         laplacian_loss_weight: float = 0.05,
#     ):
#         """
#         Reconstruction loss combining:
#           - SmoothL1 (Huber) value loss
#           - Cosine similarity over flattened feature maps
#           - Laplacian loss for spatial detail preservation

#         Args:
#             huber_beta: Transition point for SmoothL1/Huber loss
#             cosine_loss_weight: Weight for cosine similarity term (helps in feature space)
#             laplacian_loss_weight: Weight for Laplacian loss term (spatial detail)
#         """
#         super().__init__()
#         self.value_loss = torch.nn.SmoothL1Loss(beta=huber_beta, reduction="mean")
#         self.cosine_loss_weight = cosine_loss_weight
#         self.laplacian_loss_weight = laplacian_loss_weight
#         self.cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
#         self.laplacian = LaplacianLoss()

#     def forward(
#         self, graph_true: DenseGraphBatch, graph_pred: DenseGraphBatch
#     ) -> Dict[str, Any]:
#         device = graph_pred.node_features.device
#         mask = graph_true.mask.to(device) if graph_true.mask is not None else None
#         # Expected shapes: [B, N, D]
#         nodes_true_3d = graph_true.node_features.to(device)
#         nodes_pred_3d = graph_pred.node_features.to(device)

#         # Huber/SmoothL1 value loss on valid nodes
#         if mask is not None:
#             mask_exp = mask.unsqueeze(-1).expand_as(nodes_true_3d).float()
#             value_loss = torch.nn.functional.smooth_l1_loss(
#                 nodes_pred_3d * mask_exp, nodes_true_3d * mask_exp, beta=self.value_loss.beta, reduction="sum"
#             )
#             denom = mask_exp.sum().clamp_min(1.0)
#             value_loss = value_loss / denom
#         else:
#             value_loss = self.value_loss(nodes_pred_3d, nodes_true_3d)

#         total_loss = value_loss
#         loss_dict = {"huber_loss": value_loss}

#         # Cosine similarity over flattened per-sample feature vectors (mask padded nodes)
#         if mask is not None:
#             mask_exp = mask.unsqueeze(-1).expand_as(nodes_true_3d).float()
#             pred_masked = nodes_pred_3d * mask_exp
#             true_masked = nodes_true_3d * mask_exp
#         else:
#             pred_masked = nodes_pred_3d
#             true_masked = nodes_true_3d

#         pred_flat = pred_masked.flatten(1)  # [B, N*D]
#         true_flat = true_masked.flatten(1)  # [B, N*D]
#         cosine_sim = self.cosine_sim(pred_flat, true_flat).mean()
#         cosine_loss = 1.0 - cosine_sim
#         total_loss = total_loss + self.cosine_loss_weight * cosine_loss
#         loss_dict["cosine_loss"] = cosine_loss

#         # Laplacian loss on spatial grids (assumes N = H*W)
#         B, N, D = nodes_true_3d.shape
#         grid_size = int(N ** 0.5)
#         if grid_size * grid_size == N:
#             pred_grid = nodes_pred_3d.view(B, grid_size, grid_size, D).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]
#             true_grid = nodes_true_3d.view(B, grid_size, grid_size, D).permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]
#             laplacian_loss = self.laplacian(pred_grid, true_grid)
#             total_loss = total_loss + self.laplacian_loss_weight * laplacian_loss
#             loss_dict["laplacian_loss"] = laplacian_loss
#         else:
#             # Fallback: no Laplacian if grid cannot be formed
#             loss_dict["laplacian_loss"] = torch.tensor(0.0, device=device, dtype=nodes_true_3d.dtype)

#         # Return the complete loss dictionary
#         loss_dict["loss"] = total_loss
#         return loss_dict


class HuberLoss(BaseReconstructionLoss):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.smooth_l1_loss(pred, target, beta=self.beta, reduction="mean")


class CosineLoss(BaseGridReconstructionLoss):
    def __init__(self, dim: int = 1, eps: float = 1e-8):
        super().__init__()
        self.cosine_sim = torch.nn.CosineSimilarity(dim=dim, eps=eps)

    def forward(self, pred_grid: torch.Tensor, true_grid: torch.Tensor) -> torch.Tensor:
        sim = self.cosine_sim(pred_grid, true_grid)
        return 1.0 - sim.mean()


class GradientLoss(BaseGridReconstructionLoss):
    def __init__(self):
        super().__init__()

    def forward(self, pred_grid: torch.Tensor, true_grid: torch.Tensor) -> torch.Tensor:
        # Compute gradients in both directions
        pred_grad_x = pred_grid[:, :, :, 1:] - pred_grid[:, :, :, :-1]
        true_grad_x = true_grid[:, :, :, 1:] - true_grid[:, :, :, :-1]

        pred_grad_y = pred_grid[:, :, 1:, :] - pred_grid[:, :, :-1, :]
        true_grad_y = true_grid[:, :, 1:, :] - true_grid[:, :, :-1, :]

        # L1 loss on gradients (preserves sharp edges better than L2)
        gradient_loss = torch.mean(torch.abs(pred_grad_x - true_grad_x)) + torch.mean(
            torch.abs(pred_grad_y - true_grad_y)
        )

        return gradient_loss


class TVLoss(BaseGridReconstructionLoss):
    def __init__(self):
        super().__init__()

    def forward(self, pred_grid: torch.Tensor, true_grid: torch.Tensor) -> torch.Tensor:
        tv_loss = torch.mean(
            torch.abs(pred_grid[:, :, 1:, :] - pred_grid[:, :, :-1, :])
        ) + torch.mean(torch.abs(pred_grid[:, :, :, 1:] - pred_grid[:, :, :, :-1]))
        return tv_loss


class CharbonnierLoss(BaseGridReconstructionLoss):
    def __init__(self):
        super().__init__()

    def forward(
        self, pred_grid: torch.Tensor, true_grid: torch.Tensor, epsilon: float = 1e-6
    ) -> torch.Tensor:
        pred_grad_x = pred_grid[:, :, :, 1:] - pred_grid[:, :, :, :-1]
        true_grad_x = true_grid[:, :, :, 1:] - true_grid[:, :, :, :-1]

        pred_grad_y = pred_grid[:, :, 1:, :] - pred_grid[:, :, :-1, :]
        true_grad_y = true_grid[:, :, 1:, :] - true_grid[:, :, :-1, :]

        grad_diff_x = torch.sqrt((pred_grad_x - true_grad_x) ** 2 + epsilon)
        grad_diff_y = torch.sqrt((pred_grad_y - true_grad_y) ** 2 + epsilon)

        charbonnier_loss = grad_diff_x.mean() + grad_diff_y.mean()
        return charbonnier_loss


class SSIMLoss(BaseReconstructionLoss):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_loss = 1 - pytorch_msssim.ssim(pred, target, win_size=3, data_range=10.0)
        return ssim_loss


class LPIPSLoss(BaseReconstructionLoss):
    def __init__(self):
        super().__init__()
        self.lpips = lpips.LPIPS(net="vgg")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lpips_loss = self.lpips(pred, target)
        return lpips_loss


class HuberGradientLoss(BaseGridReconstructionLoss):
    def __init__(self):
        super().__init__()

    def forward(self, pred_grid: torch.Tensor, true_grid: torch.Tensor) -> torch.Tensor:
        # x-gradient (width)
        pred_grad_x = pred_grid[:, :, :, 1:] - pred_grid[:, :, :, :-1]
        true_grad_x = true_grid[:, :, :, 1:] - true_grid[:, :, :, :-1]

        # y-gradient (height)
        pred_grad_y = pred_grid[:, :, 1:, :] - pred_grid[:, :, :-1, :]
        true_grad_y = true_grid[:, :, 1:, :] - true_grid[:, :, :-1, :]
        grad_loss_x = torch.nn.functional.huber_loss(pred_grad_x, true_grad_x, delta=1.0)
        grad_loss_y = torch.nn.functional.huber_loss(pred_grad_y, true_grad_y, delta=1.0)
        gradient_loss = grad_loss_x + grad_loss_y
        return gradient_loss


class LaplacianLoss(BaseGridReconstructionLoss):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer("lap_kernel", kernel.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, c, _, _ = pred.shape
        kernel = self.lap_kernel.repeat(c, 1, 3, 3).to(dtype=pred.dtype, device=pred.device)
        lap_pred = F.conv2d(pred, kernel, padding=1, groups=c)
        lap_true = F.conv2d(target, kernel, padding=1, groups=c)
        return F.l1_loss(lap_pred, lap_true)


class MAELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_loss = L1Loss()  # BCEWithLogitsLoss() #MSE

    def forward(self, graph_true: DenseGraphBatch, graph_pred: DenseGraphBatch) -> torch.Tensor:
        # Use the mask to identify valid nodes
        device = graph_pred.node_features.device
        mask = graph_true.mask
        if mask is None:
            b, n, _ = graph_true.node_features.shape
            mask = torch.ones(b, n, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device)
        # Extract the node features for the true and predicted graphs, filtered
        # by the mask
        nodes_true = graph_true.node_features.to(device)
        nodes_true = nodes_true[mask]
        nodes_pred = graph_pred.node_features[mask]

        # Compute the node-based loss
        loss = self.node_loss(input=nodes_pred, target=nodes_true)

        return loss


class MSEGraphLoss(BaseReconstructionLoss):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(pred, target, reduction="mean")


class MSEGridLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_loss = MSELoss()

    def forward(self, graph_true: DenseGraphBatch, graph_pred: DenseGraphBatch) -> torch.Tensor:
        # Use the mask to identify valid nodes
        device = graph_pred.node_features.device
        mask = graph_true.mask
        if mask is None:
            b, n, _ = graph_true.node_features.shape
            mask = torch.ones(b, n, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device)
        # Extract the node features for the true and predicted graphs, filtered
        # by the mask
        nodes_true = graph_true.node_features.to(device)
        nodes_true = nodes_true[mask]
        nodes_pred = graph_pred.node_features[mask]

        # Compute the node-based loss
        loss = self.node_loss(input=nodes_pred, target=nodes_true)

        return loss


class CosineSimilarityLoss(BaseReconstructionLoss):
    def __init__(self, return_as_loss: bool = True):
        """
        Args:
            return_as_loss: If True, returns (1 - cosine_similarity) so lower is better.
                          If False, returns cosine_similarity directly (higher is better).
        """
        super().__init__()
        # dim = 1 is channel reconstrtion
        # dim = -1 is node reconstruction
        self.node_loss = CosineSimilarity(dim=-1, eps=1e-8)
        self.return_as_loss = return_as_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Use the mask to identify valid nodes
        similarity = self.node_loss(pred.flatten(1), target.flatten(1)).mean()

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
        if mask is None:
            b, n, _ = graph_true.node_features.shape
            mask = torch.ones(b, n, dtype=torch.bool, device=device)
        else:
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
        """KLD Loss with optional free bits to prevent posterior collapse.

        Args:
            normalize_by_latent_dim: If True, average over latent dims instead of sum
            free_bits: Free bits threshold. KLD below this per dimension is not penalized
                      (dead zone — no gradient below threshold). Set to 0.0 to always
                      have gradient. Typical values: 0.0 (disabled), 0.5, 1.0, 2.0
        """
        super().__init__()
        self.normalize_by_latent_dim = normalize_by_latent_dim
        self.free_bits = free_bits

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Compute KL in fp32 and clamp log-variance to avoid numerical overflow under AMP.
        with torch.autocast(device_type="cuda", enabled=False):
            mu32 = mu.float()
            logvar32 = logvar.float().clamp(-10.0, 10.0)
            # KL divergence: -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
            kld_per_dim = -0.5 * (1 + logvar32 - mu32.pow(2) - logvar32.exp())

        if self.free_bits > 0:
            # Free bits: do not penalize the first `free_bits` nats of KL per latent dimension.
            # Only KL above that threshold contributes to the loss and receives gradient.
            kld_per_dim = torch.relu(kld_per_dim - self.free_bits)

        if self.normalize_by_latent_dim:
            # Average over latent dimensions, then average over batch
            loss = torch.mean(kld_per_dim)
        else:
            # Sum over latent dimensions, then average over batch (original behavior)
            loss = torch.sum(kld_per_dim, dim=1).mean()
        # Guard against NaNs/Infs that can appear rarely with extreme values
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)
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
        n = features.shape[0]
        samples_per_group = 1 + self.num_aug_per_sample
        original_batch_size = n // samples_per_group  # number of original images
        labels = torch.cat(
            [torch.arange(original_batch_size) for _ in range(samples_per_group)], dim=0
        )
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).bool()
        label_matrix = label_matrix.to(features.device)
        # # Step 4: Similarity matrix
        sim = torch.matmul(features, features.T)  # [N, N]
        sim = sim / self.temperature

        # Mask out self-similarity
        mask = torch.eye(n, dtype=torch.bool, device=features.device)
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
    def __init__(self, num_permutations: int = 8):
        super().__init__()
        self.num_permutations = num_permutations

    def forward(self, probs: torch.Tensor = None):
        if probs is None:
            return torch.tensor(0.0)
        # Batch layout: rows 0..B-1 = view 0, rows B..2B-1 = view 1, ..., rows 7B..8B-1 = view 7.
        # For each image, average the 8 view predictions — this average should be uniform
        # (each view predicted a different class). This directly penalises the failure mode
        # where all 8 views of the same image collapse to the same class, which the old
        # batch-average entropy could not detect.
        total = probs.shape[0]
        batch_size = total // self.num_permutations
        # [num_perms, B, num_classes] → mean over views → [B, num_classes]
        avg_per_image = probs.view(self.num_permutations, batch_size, self.num_permutations).mean(dim=0)
        log_avg = torch.log(avg_per_image + 1e-12)
        entropy_per_image = -(avg_per_image * log_avg).sum(dim=-1)  # [B]
        max_entropy = torch.log(torch.tensor(self.num_permutations, dtype=probs.dtype, device=probs.device))
        return (max_entropy - entropy_per_image).mean()


class PermutaionMatrixLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def entropy(p: torch.Tensor, axis: int, normalize: bool = True, eps=10e-12) -> torch.Tensor:
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
