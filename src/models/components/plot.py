from collections import defaultdict

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from plotly.graph_objs import Figure as PlotlyFigure
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


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
        a.size(1) ** 0.5,
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
        temp = images[idx * n_cols + n_rows - idx : (idx + 1) * n_cols + n_rows - 1 - idx]
        new_images.extend(temp)
    return np.array(new_images)


def plot_images_all_perm(
    images: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> figure.Figure | None:
    if not len(images):
        return None
    if images.shape[0] < n_rows * n_cols:
        raise ValueError("Not enough images to fill the grid.")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()  # Flatten in case of multiple rows
    # new_images = reshape_images_array(images, n_rows, n_cols)

    for idx, img in enumerate(images):
        ax = axes[idx]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Image {idx}")
        ax.axis("off")

    return fig


def plot_barchart_from_dict(data: dict[str, float], title: str) -> figure.Figure:
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(list(data.keys()), list(data.values()), color="skyblue")
    ax.set_xlabel("Keys")
    ax.set_ylabel("Values")
    ax.set_title(title)

    return fig


def plot_pca(
    images: np.ndarray,
    targets: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> figure.Figure:
    # new_images = reshape_images_array(images, n_rows, n_cols)
    counter: dict[int, int] = defaultdict(int)
    labels_list: list[str] = []
    for val in targets:
        counter[int(val)] += 1
        labels_list.append(f"{val.item()}-{counter[int(val)]}")
    labels = np.repeat(labels_list, n_cols)

    batch_flat = images.reshape(images.shape[0], -1)
    # Step 2: Run PCA
    pca = PCA(n_components=2)  # choose desired number of components
    batch_pca = pca.fit_transform(batch_flat)
    scaler = StandardScaler()
    batch_pca_scaled = scaler.fit_transform(batch_pca)  # shape: [64, 2]

    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is not None:
        classes = np.unique(labels)
        cmap = plt.cm.get_cmap("tab10", len(classes))

        for idx, cls in enumerate(classes):
            mask = labels == cls
            ax.scatter(
                batch_pca_scaled[mask, 0],
                batch_pca_scaled[mask, 1],
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


def plot_pca_plotly(
    images: np.ndarray,
    targets: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> PlotlyFigure:
    # Reshape images into grid format (assuming you already have this helper)
    # new_images = reshape_images_array(images, n_rows, n_cols)
    counter: dict[int, int] = defaultdict(int)
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
    labels_list: list[str] = []

    for val in targets:
        counter[int(val)] += 1
        labels_list.append(f"{val.item()}-{counter[int(val)]}")
    labels = np.repeat(labels_list, n_cols)

    # Flatten images for PCA
    batch_flat = images.reshape(images.shape[0], -1)
    pca = PCA(n_components=2)
    batch_pca = pca.fit_transform(batch_flat)

    # Create DataFrame for Plotly
    df = pd.DataFrame(batch_pca, columns=["PC1", "PC2"])
    df["Label"] = labels
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


def plot_silhouette(
    images: np.ndarray,
    k_range: range | None = None,
) -> tuple[int, figure.Figure]:
    if k_range is None:
        k_range = range(2, 11)
    best_k = 0
    best_score = -1
    silhouette_scores = []

    batch_flat = images.reshape(images.shape[0], -1)
    scaler = StandardScaler()
    batch_flat_scaled = scaler.fit_transform(batch_flat)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(batch_flat_scaled)
        score = silhouette_score(batch_flat_scaled, labels)
        silhouette_scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k
    # Plot and capture figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_range, silhouette_scores, marker="o")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Average Silhouette Score")
    ax.set_title(f"Silhouette Score vs. k (Best k={best_k}, score={best_score:.4f})")
    plt.tight_layout()

    return best_k, fig


def plot_inter_silhouette(images: np.ndarray, k: int) -> figure.Figure:
    # Assuming X is your unlabelled data of shape [n_samples, n_features]
    # choose the number of clusters you want to evaluate

    batch_flat = images.reshape(images.shape[0], -1)
    scaler = StandardScaler()
    batch_flat_scaled = scaler.fit_transform(batch_flat)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(batch_flat_scaled)

    score = silhouette_score(batch_flat_scaled, labels)
    sample_scores = silhouette_samples(batch_flat_scaled, labels)

    y_lower = 10
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(k):
        ith_cluster_silhouette_values = sample_scores[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        cmap = plt.get_cmap("nipy_spectral")
        color = cmap(float(i) / k)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=score, color="red", linestyle="--")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")
    ax.set_title(f"Silhouette Plot (avg score: {score:.3f})")
    return fig
