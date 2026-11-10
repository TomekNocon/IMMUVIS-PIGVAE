"""Reconstruction, KL, permutation and contrastive losses.

Node-level losses take ``[B, N, D]`` node features and subclass :class:`BaseReconstructionLoss`;
grid-level losses take ``[B, D, H, W]`` grids and subclass :class:`BaseGridReconstructionLoss`.
:class:`GraphReconstructionLoss` reshapes the nodes to a grid where needed and sums three
weighted terms.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity, L1Loss, MSELoss

from src.data.components.graphs_datamodules import DenseGraphBatch


class BaseGridReconstructionLoss(torch.nn.Module):
    """Loss on ``[B, D, H, W]`` grids."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BaseReconstructionLoss(torch.nn.Module):
    """Loss on ``[B, N, D]`` node features."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _masked_nodes(
    graph_true: DenseGraphBatch, graph_pred: DenseGraphBatch
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(true, pred)`` node features of the valid (mask=True) nodes, flattened to ``[M, D]``."""
    device = graph_pred.node_features.device
    mask = graph_true.mask
    if mask is None:
        b, n, _ = graph_true.node_features.shape
        mask = torch.ones(b, n, dtype=torch.bool, device=device)
    else:
        mask = mask.to(device)
    return graph_true.node_features.to(device)[mask], graph_pred.node_features[mask]


class GraphReconstructionLoss(torch.nn.Module):
    """Weighted sum ``alpha * L_alpha + beta * L_beta + gamma * L_gamma``.

    Each term is evaluated on the node features or on the ``sqrt(N) x sqrt(N)`` grid,
    depending on its base class. Returns the weighted terms (as floats, for logging) and the
    differentiable total under ``"loss"``.
    """

    def __init__(
        self,
        loss_alpha: BaseReconstructionLoss | BaseGridReconstructionLoss,
        loss_beta: BaseReconstructionLoss | BaseGridReconstructionLoss,
        loss_gamma: BaseReconstructionLoss | BaseGridReconstructionLoss,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.losses = torch.nn.ModuleDict({
            "alpha_recon": loss_alpha,
            "beta_recon": loss_beta,
            "gamma_recon": loss_gamma,
        })
        self.weights = {"alpha_recon": alpha, "beta_recon": beta, "gamma_recon": gamma}

    def forward(self, graph_true: DenseGraphBatch, graph_pred: DenseGraphBatch) -> dict[str, Any]:
        device = graph_pred.node_features.device
        true = graph_true.node_features.to(device)
        pred = graph_pred.node_features.to(device)

        b, n, d = graph_true.node_features.shape
        side = math.isqrt(int(n))
        true_grid = true.view(b, side, side, d).permute(0, 3, 1, 2)
        pred_grid = pred.view(b, side, side, d).permute(0, 3, 1, 2)

        out: dict[str, Any] = {}
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


# --------------------------------------------------------------------------- node-level terms


class HuberLoss(BaseReconstructionLoss):
    """Smooth-L1 (Huber) loss with transition point ``beta``."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=self.beta, reduction="mean")


class MSEGraphLoss(BaseReconstructionLoss):
    """Mean squared error over all node features."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction="mean")


class CosineSimilarityLoss(BaseReconstructionLoss):
    """Per-node cosine distance between predicted and true feature vectors.

    Args:
        return_as_loss: Return ``1 - cosine`` (0 = perfect, 2 = worst); otherwise the cosine.
    """

    def __init__(self, return_as_loss: bool = True):
        super().__init__()
        self.node_loss = CosineSimilarity(dim=-1, eps=1e-8)
        self.return_as_loss = return_as_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Cosine of each node's [D] vector, averaged over batch and nodes. A single cosine over
        # the flattened sample would be dominated by the high-magnitude nodes.
        similarity = self.node_loss(pred, target).mean()
        return 1 - similarity if self.return_as_loss else similarity


class SSIMLoss(BaseReconstructionLoss):
    """``1 - SSIM`` (3x3 window, data range 10)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        import pytorch_msssim

        return 1 - pytorch_msssim.ssim(pred, target, win_size=3, data_range=10.0)


class LPIPSLoss(BaseReconstructionLoss):
    """LPIPS perceptual distance with a VGG backbone."""

    def __init__(self):
        super().__init__()
        import lpips

        self.lpips = lpips.LPIPS(net="vgg")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lpips(pred, target)


# --------------------------------------------------------------------------- grid-level terms


def _grid_gradients(
    grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward differences along width (x) and height (y) of a ``[B, D, H, W]`` grid."""
    return grid[:, :, :, 1:] - grid[:, :, :, :-1], grid[:, :, 1:, :] - grid[:, :, :-1, :]


class CosineLoss(BaseGridReconstructionLoss):
    """``1 - cosine`` along ``dim`` of the grids (default: the channel dimension)."""

    def __init__(self, dim: int = 1, eps: float = 1e-8):
        super().__init__()
        self.cosine_sim = torch.nn.CosineSimilarity(dim=dim, eps=eps)

    def forward(self, pred_grid: torch.Tensor, true_grid: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.cosine_sim(pred_grid, true_grid).mean()


class GradientLoss(BaseGridReconstructionLoss):
    """L1 distance between the spatial gradients of the grids (preserves sharp edges)."""

    def forward(self, pred_grid: torch.Tensor, true_grid: torch.Tensor) -> torch.Tensor:
        pred_x, pred_y = _grid_gradients(pred_grid)
        true_x, true_y = _grid_gradients(true_grid)
        return torch.mean(torch.abs(pred_x - true_x)) + torch.mean(torch.abs(pred_y - true_y))


class HuberGradientLoss(BaseGridReconstructionLoss):
    """Huber distance between the spatial gradients of the grids."""

    def forward(self, pred_grid: torch.Tensor, true_grid: torch.Tensor) -> torch.Tensor:
        pred_x, pred_y = _grid_gradients(pred_grid)
        true_x, true_y = _grid_gradients(true_grid)
        return F.huber_loss(pred_x, true_x, delta=1.0) + F.huber_loss(pred_y, true_y, delta=1.0)


class CharbonnierLoss(BaseGridReconstructionLoss):
    """Charbonnier (smooth L1) distance between the spatial gradients of the grids."""

    def forward(
        self, pred_grid: torch.Tensor, true_grid: torch.Tensor, epsilon: float = 1e-6
    ) -> torch.Tensor:
        pred_x, pred_y = _grid_gradients(pred_grid)
        true_x, true_y = _grid_gradients(true_grid)
        diff_x = torch.sqrt((pred_x - true_x) ** 2 + epsilon)
        diff_y = torch.sqrt((pred_y - true_y) ** 2 + epsilon)
        return diff_x.mean() + diff_y.mean()


class TVLoss(BaseGridReconstructionLoss):
    """Total variation of the predicted grid (the target is ignored)."""

    def forward(self, pred_grid: torch.Tensor, true_grid: torch.Tensor) -> torch.Tensor:
        grad_x, grad_y = _grid_gradients(pred_grid)
        return torch.mean(torch.abs(grad_y)) + torch.mean(torch.abs(grad_x))


class LaplacianLoss(BaseGridReconstructionLoss):
    """L1 distance between the per-channel discrete Laplacians of the grids."""

    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer("lap_kernel", kernel.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.shape[1]
        kernel = self.lap_kernel.repeat(channels, 1, 1, 1).to(dtype=pred.dtype, device=pred.device)
        lap_pred = F.conv2d(pred, kernel, padding=1, groups=channels)
        lap_true = F.conv2d(target, kernel, padding=1, groups=channels)
        return F.l1_loss(lap_pred, lap_true)


# --------------------------------------------------------------------------- logged metrics


class MAELoss(torch.nn.Module):
    """Mean absolute error over the valid nodes."""

    def __init__(self):
        super().__init__()
        self.node_loss = L1Loss()

    def forward(self, graph_true: DenseGraphBatch, graph_pred: DenseGraphBatch) -> torch.Tensor:
        nodes_true, nodes_pred = _masked_nodes(graph_true, graph_pred)
        return self.node_loss(input=nodes_pred, target=nodes_true)


class MSEGridLoss(torch.nn.Module):
    """Mean squared error over the valid nodes."""

    def __init__(self):
        super().__init__()
        self.node_loss = MSELoss()

    def forward(self, graph_true: DenseGraphBatch, graph_pred: DenseGraphBatch) -> torch.Tensor:
        nodes_true, nodes_pred = _masked_nodes(graph_true, graph_pred)
        return self.node_loss(input=nodes_pred, target=nodes_true)


class SignalToNoiseRatioLoss(torch.nn.Module):
    """Reconstruction SNR in dB over the valid nodes (higher is better; logged, not optimised)."""

    def forward(
        self, graph_true: DenseGraphBatch, graph_pred: DenseGraphBatch, eps: float = 1e-8
    ) -> torch.Tensor:
        nodes_true, nodes_pred = _masked_nodes(graph_true, graph_pred)
        signal_power = torch.mean(nodes_true**2)
        noise_power = torch.mean((nodes_pred - nodes_true) ** 2)
        return 10 * torch.log10(signal_power / (noise_power + eps))


# --------------------------------------------------------------------------- latent terms


class KLDLoss(torch.nn.Module):
    """KL divergence to a standard normal, with optional free bits.

    Args:
        normalize_by_latent_dim: Average over latent dimensions (else sum, then batch mean).
        free_bits: Per-dimension KL (nats) that is not penalised; below it there is no
            gradient. 0 disables it.
    """

    def __init__(self, normalize_by_latent_dim: bool = True, free_bits: float = 0.0):
        super().__init__()
        self.normalize_by_latent_dim = normalize_by_latent_dim
        self.free_bits = free_bits

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # fp32 with a clamped log-variance to avoid overflow under AMP.
        with torch.autocast(device_type="cuda", enabled=False):
            mu32 = mu.float()
            logvar32 = logvar.float().clamp(-10.0, 10.0)
            kld_per_dim = -0.5 * (1 + logvar32 - mu32.pow(2) - logvar32.exp())

        if self.free_bits > 0:
            kld_per_dim = torch.relu(kld_per_dim - self.free_bits)

        if self.normalize_by_latent_dim:
            loss = torch.mean(kld_per_dim)
        else:
            loss = torch.sum(kld_per_dim, dim=1).mean()
        return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)


class ContrastiveLoss(torch.nn.Module):
    """SimCLR-style loss treating all ``num_aug_per_sample`` views of an image as positives.

    Expects the batch as ``num_aug_per_sample`` stacked blocks of the same images.
    """

    def __init__(self, temperature: float = 0.07, num_aug_per_sample: int = 8):
        super().__init__()
        self.temperature = temperature
        self.num_aug_per_sample = num_aug_per_sample - 1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        n = features.shape[0]
        samples_per_group = 1 + self.num_aug_per_sample
        labels = torch.arange(n // samples_per_group).repeat(samples_per_group)
        same_image = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(features.device)

        sim = torch.matmul(features, features.T) / self.temperature  # [N, N]
        diagonal = torch.eye(n, dtype=torch.bool, device=features.device)
        sim.masked_fill_(diagonal, float("-inf"))

        pos_term = torch.logsumexp(sim.masked_fill(~(same_image & ~diagonal), float("-inf")), dim=1)
        neg_term = torch.logsumexp(sim.masked_fill(same_image, float("-inf")), dim=1)
        return (neg_term - pos_term).mean()


class PermutationLoss(torch.nn.Module):
    """Per-image view-diversity loss for the permuter: ``log K - H(mean_views(p))``.

    The batch holds ``num_permutations`` stacked view blocks. Averaging each image's view
    predictions should give a uniform distribution when every view is assigned a different
    D4 class; this penalises all views of one image collapsing onto the same class.
    """

    def __init__(self, num_permutations: int = 8):
        super().__init__()
        self.num_permutations = num_permutations

    def forward(self, probs: torch.Tensor | None = None) -> torch.Tensor:
        if probs is None:
            return torch.tensor(0.0)
        k = self.num_permutations
        avg_per_image = probs.view(k, probs.shape[0] // k, k).mean(dim=0)  # [B, K]
        entropy_per_image = -(avg_per_image * torch.log(avg_per_image + 1e-12)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(k, dtype=probs.dtype, device=probs.device))
        return (max_entropy - entropy_per_image).mean()


class PermutationMatrixLoss(torch.nn.Module):
    """Row + column entropy of soft permutation matrices (low when they are near-hard)."""

    @staticmethod
    def entropy(
        p: torch.Tensor, axis: int, normalize: bool = True, eps: float = 1e-11
    ) -> torch.Tensor:
        if normalize:
            p = p / (p.sum(axis=axis, keepdim=True) + eps)
        return -torch.sum(p * torch.clamp_min(torch.log(p), -100), axis=axis)

    def forward(self, perm: torch.Tensor | None, eps: float = 1e-7) -> torch.Tensor:
        if perm is None:
            return torch.tensor(0.0)
        perm = perm + eps
        entropy_col = self.entropy(perm, axis=1, normalize=False)
        entropy_row = self.entropy(perm, axis=2, normalize=False)
        return entropy_col.mean() + entropy_row.mean()
