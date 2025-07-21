from typing import Dict
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np


def batch_augmented_indices(
    batch_size: int, num_permutations: int, n_examples: int
) -> np.ndarray:
    indices = []
    base_idx = np.arange(num_permutations) * batch_size
    for i in range(n_examples):
        indices.append(base_idx + i)
    return np.stack(indices, axis=0).flatten()


def to_original_orientation(
    predictions: torch.Tensor, num_permutations: int = 8
) -> torch.Tensor:
    """
    Inverts the 8 augmentations (original, hflip, rot90, rot90_hflip, rot180, rot180_hflip, rot270, rot270_hflip)
    back to the original orientation.

    Args:
        predictions: [8 * batch_size, C, H, W]
        batch_size: int

    Returns:
        torch.Tensor: [8 * batch_size, C, H, W] with all tensors aligned to original orientation.
    """

    inverted = []
    augmented_batch_size = predictions.shape[0]
    batch_size = augmented_batch_size // num_permutations
    for idx in range(num_permutations):
        preds = predictions[idx * batch_size : (idx + 1) * batch_size]

        # Determine if hflip and rotation should be applied
        do_hflip = idx % 2 == 1
        rotation_deg = -90 * ((idx // 2) % 4)  # 0, -90, -180, -270

        if do_hflip:
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
) -> Dict:
    transform_losses = {}
    transform_names = [
        "original",
        "hflip",
        "rot90",
        "rot90_hflip",
        "rot180",
        "rot180_hflip",
        "rot270",
        "rot270_hflip",
    ]

    for idx, name in zip(range(num_permutations), transform_names):
        y = ground_truth[idx * batch_size : (idx + 1) * batch_size]
        y_hat = predictions[idx * batch_size : (idx + 1) * batch_size]
        loss = F.mse_loss(y, y_hat).item()
        transform_losses[name] = loss

    return transform_losses
