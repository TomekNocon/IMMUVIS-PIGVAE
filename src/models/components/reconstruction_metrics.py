"""Per-view reconstruction metrics for batches laid out as ``num_permutations`` stacked D4 views."""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

TRANSFORM_NAMES = [
    "original",
    "hflip",
    "rot90",
    "rot90_hflip",
    "rot180",
    "rot180_hflip",
    "rot270",
    "rot270_hflip",
]


def batch_augmented_indices(batch_size: int, num_permutations: int, n_examples: int) -> np.ndarray:
    """Indices of all views of the first ``n_examples`` samples, grouped per sample."""
    base_idx = np.arange(num_permutations) * batch_size
    return np.stack([base_idx + i for i in range(n_examples)], axis=0).flatten()


def to_original_orientation(predictions: torch.Tensor, num_permutations: int = 8) -> torch.Tensor:
    """Undo the D4 augmentation of each view block, aligning all views to the original.

    Assumes the view order of ``TRANSFORM_NAMES`` (original, hflip, rot90, rot90_hflip, ...).

    Args:
        predictions: ``[num_permutations * B, C, H, W]`` images, one block of ``B`` per view.
        num_permutations: Number of stacked views.

    Returns:
        ``[num_permutations * B, C, H, W]`` with every view rotated/flipped back.
    """
    batch_size = predictions.shape[0] // num_permutations
    inverted = []
    for idx in range(num_permutations):
        preds = predictions[idx * batch_size : (idx + 1) * batch_size]
        rotation_deg = -90 * ((idx // 2) % 4)  # 0, -90, -180, -270
        if idx % 2 == 1:
            preds = TF.hflip(preds)
        if rotation_deg != 0:
            preds = TF.rotate(preds, rotation_deg)
        inverted.append(preds)
    return torch.cat(inverted, dim=0)


def mse_per_transform(
    ground_truth: torch.Tensor,
    predictions: torch.Tensor,
    batch_size: int,
    num_permutations: int,
) -> dict[str, float]:
    """MSE of each view block, keyed by transform name."""
    losses = {}
    for idx, name in zip(range(num_permutations), TRANSFORM_NAMES, strict=False):
        y = ground_truth[idx * batch_size : (idx + 1) * batch_size]
        y_hat = predictions[idx * batch_size : (idx + 1) * batch_size]
        losses[name] = F.mse_loss(y, y_hat).item()
    return losses
