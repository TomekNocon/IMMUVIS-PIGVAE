"""Laplacian-eigenvector positional embeddings of the 2D node grid."""

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import SpectralEmbedding


class CustomSpectralEmbedding(nn.Module):
    """All grid-Laplacian eigenvectors (NumPy ``eigh``), linearly projected to ``d_model``."""

    def __init__(self, d_model: int, grid_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.G = nx.grid_2d_graph(grid_size, grid_size)
        self.A = nx.to_numpy_array(self.G)
        self.D = np.diag(self.A.sum(axis=0))
        self.L = self.D - self.A
        sorted_eigenvecs = self.compute_eigen(self.L)
        self.fc1 = nn.Linear(grid_size**2, d_model)

        self.register_buffer("sorted_eigenvecs", sorted_eigenvecs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _ = x.shape
        embedding = self.sorted_eigenvecs.transpose(-2, -1)
        embedding = self.fc1(embedding)
        embedding = torch.tile(embedding, (batch, 1, 1))
        x = x + embedding
        return self.dropout(x)

    def compute_eigen(self, laplacian: np.ndarray) -> torch.Tensor:
        eigenvals, eigenvecs = np.linalg.eigh(laplacian)
        sorted_eigenvecs = eigenvecs[:, np.argsort(eigenvals)]
        sorted_eigenvecs = torch.tensor(sorted_eigenvecs, dtype=torch.float32)
        return sorted_eigenvecs


class NetworkXSpectralEmbedding(nn.Module):
    """``networkx.spectral_layout`` coordinates of the grid, added to the node features."""

    def __init__(self, d_model: int, grid_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.G = nx.grid_2d_graph(grid_size, grid_size)
        sorted_eigenvecs = torch.tensor(
            list(nx.spectral_layout(self.G, dim=d_model).values()), dtype=torch.float32
        )

        self.register_buffer("sorted_eigenvecs", sorted_eigenvecs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _ = x.shape
        embedding = torch.tile(self.sorted_eigenvecs, (batch, 1, 1))
        x = x.to(embedding.device) + embedding
        return self.dropout(x)


class SklearnSpectralEmbedding(nn.Module):
    """Spectral grid embedding (scikit-learn) added to layer-normalised node content.

    Used by the permuter: the content is normalised to unit scale before the embedding is
    added, so the positional signal cannot drown out (or be drowned by) the node features.
    """

    def __init__(
        self,
        n_components: int,
        d_model: int,
        grid_size: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        adjacency = nx.to_numpy_array(nx.grid_2d_graph(grid_size, grid_size))
        transformation = SpectralEmbedding(
            n_components=n_components, affinity="precomputed", **kwargs
        )
        sorted_eigenvecs = torch.tensor(
            transformation.fit_transform(adjacency), dtype=torch.float32
        )
        self.register_buffer("sorted_eigenvecs", sorted_eigenvecs)  # [N, n_components]

        self.proj = nn.Linear(n_components, d_model)
        # elementwise_affine=False: proj learns direction, not scale (prevents unbounded growth).
        self.proj_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.to_project = d_model != n_components
        # Normalise content to unit scale before adding the embedding (std~1.1), which would
        # otherwise drown out the learned content features (std~0.15) early in training.
        self.content_norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _ = x.shape
        embedding = self.sorted_eigenvecs.unsqueeze(0).expand(batch, -1, -1)
        if self.to_project:
            embedding = self.proj_norm(self.proj(embedding))

        # Dropout before adding (not after) keeps the scale of x stable.
        embedding = self.dropout(embedding)
        return self.content_norm(x) + embedding
