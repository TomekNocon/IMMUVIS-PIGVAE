import matplotlib.pyplot as plt
import matplotlib.figure as figure
from typing import Optional
import torch
import numpy as np
from sklearn.decomposition import PCA
from collections import defaultdict
import plotly.express as px
import pandas as pd


def restore_tensor(
    a: torch.Tensor,
    bs: int,
    c: int,
    h: int,
    w: int,
    patch_size: int,
    to_tensor: bool = False,
) -> torch.Tensor:
    # Step 1: Reshape a to match the patch grid layout
    a = a.view(bs, -1, c, patch_size, patch_size)
    # a -> (B, num_patches, C, patch_size, patch_size)

    # Step 2: Reshape back to (B, C, H, W) by folding the patches
    # Calculate the grid size (L) which is the number of patches in each row and column
    grid_size = int(
        a.size(1) ** 0.5
    )  # Assumes square grid (height = width for the patches)

    # Unfold back into the original image size
    a = a.view(bs, grid_size, grid_size, c, patch_size, patch_size)
    # a -> (B, grid_size, grid_size, C, patch_size, patch_size)

    # Step 3: Permute and reshape to get back to the image format (B, C, H, W)
    x_reconstructed = a.permute(0, 3, 1, 4, 2, 5).contiguous()
    # x_reconstructed -> (B, C, grid_size, patch_size, grid_size, patch_size)

    x_reconstructed = x_reconstructed.view(bs, c, h, w)
    # x_reconstructed -> (B, C, H, W)
    if to_tensor:
        x_reconstructed = torch.tensor(x_reconstructed)

    return x_reconstructed


# def plot_images(images: np.array, n_images: int) -> plt.Figure:
#     fig, axes = plt.subplots(1, n_images, figsize=(15, 6))
#     for idx, ax in enumerate(axes.flat):
#         ax.imshow(images[idx], cmap="gray")
#         ax.set_title(f"Image {idx}")
#         ax.axis("off")
#     return fig


def reshape_images_array(images: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    new_images = []
    for idx in range(n_rows):
        new_images.append(images[idx])
        temp = images[
            idx * n_cols + n_rows - idx : (idx + 1) * n_cols + n_rows - 1 - idx
        ]
        for img in temp:
            new_images.append(img)
    return np.array(new_images)


def plot_images_all_perm(
    images: np.ndarray, n_rows: int, n_cols: int
) -> Optional[figure.Figure]:
    if not len(images):
        return
    assert images.shape[0] >= n_rows * n_cols, "Not enough images to fill the grid."

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()  # Flatten in case of multiple rows
    # new_images = reshape_images_array(images, n_rows, n_cols)

    for idx, img in enumerate(images):
        ax = axes[idx]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Image {idx}")
        ax.axis("off")

    return fig


def plot_pca(
    images: np.ndarray, targets: np.ndarray, n_rows: int, n_cols: int
) -> figure.Figure:
    # new_images = reshape_images_array(images, n_rows, n_cols)
    counter = defaultdict(int)
    new_targets = []
    for val in targets:
        counter[int(val)] += 1
        new_targets.append(f"{val.item()}-{counter[int(val)]}")
    new_targets = np.repeat(new_targets, n_cols)

    batch_flat = images.reshape(images.shape[0], -1)
    # Step 2: Run PCA
    pca = PCA(n_components=2)  # choose desired number of components
    batch_pca = pca.fit_transform(batch_flat)  # shape: [64, 2]

    fig, ax = plt.subplots(figsize=(8, 6))
    if new_targets is not None:
        classes = np.unique(new_targets)
        cmap = plt.cm.get_cmap("tab10", len(classes))

        for idx, cls in enumerate(classes):
            mask = new_targets == cls
            ax.scatter(
                batch_pca[mask, 0],
                batch_pca[mask, 1],
                label=str(cls),
                color=cmap(idx),
                edgecolors="k",
            )
        ax.legend(title="Class")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("PCA of Image Batch")
        ax.grid(True)

    return fig


def plot_pca_plotly(images: np.ndarray, targets: np.ndarray, n_rows: int, n_cols: int):
    # Reshape images into grid format (assuming you already have this helper)
    # new_images = reshape_images_array(images, n_rows, n_cols)
    counter = defaultdict(int)
    rotations = [
        "identity",
        "hflip",
        "90",
        "90-hflip",
        "180",
        "180-hflip",
        "270",
        "270-hflip",
    ] * n_rows
    new_targets = []

    for val in targets:
        counter[int(val)] += 1
        new_targets.append(f"{val.item()}-{counter[int(val)]}")
    new_targets = np.repeat(new_targets, n_cols)

    # Flatten images for PCA
    batch_flat = images.reshape(images.shape[0], -1)
    pca = PCA(n_components=2)
    batch_pca = pca.fit_transform(batch_flat)

    # Create DataFrame for Plotly
    df = pd.DataFrame(batch_pca, columns=["PC1", "PC2"])
    df["Label"] = new_targets
    df["Rotations"] = rotations

    # Plotly scatter plot
    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="Label",
        title="PCA of Image Batch",
        labels={"Label": "Target"},
        opacity=0.7,
        hover_data=["Rotations"],
    )
    return fig
