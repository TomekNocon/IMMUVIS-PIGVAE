import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def restore_tensor(
    a: np.array, bs: int, c: int, h: int, w: int, patch_size: int
) -> np.array:
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

    return x_reconstructed


def plot_images(images: np.array, n_images: int) -> plt.Figure:
    fig, axes = plt.subplots(1, n_images, figsize=(15, 6))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx], cmap="gray")
        ax.set_title(f"Image {idx}")
        ax.axis("off")
    return fig


def plot_pca(images: np.array) -> plt.Figure:
    batch_flat = images.reshape(images.shape[0], -1)
    # Step 2: Run PCA
    pca = PCA(n_components=2)  # choose desired number of components
    batch_pca = pca.fit_transform(batch_flat)  # shape: [64, 2]

    # Create plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(batch_pca[:, 0], batch_pca[:, 1], c="skyblue", edgecolors="k")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA of Image Batch")
    ax.grid(True)

    return fig
