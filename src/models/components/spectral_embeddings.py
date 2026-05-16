import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import SpectralEmbedding


class CustomSpectralEmbedding(nn.Module):
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
        embbeding = self.sorted_eigenvecs.transpose(-2, -1)
        embbeding = self.fc1(embbeding)
        embbeding = torch.tile(embbeding, (batch, 1, 1))
        x = x + embbeding
        return self.dropout(x)

    def compute_eigen(self, laplacian: np.array) -> torch.Tensor:
        eigenvals, eigenvecs = np.linalg.eigh(laplacian)
        sorted_eigenvecs = eigenvecs[:, np.argsort(eigenvals)]
        sorted_eigenvecs = torch.tensor(sorted_eigenvecs, dtype=torch.float32)
        return sorted_eigenvecs


class NetworkXSpectralEmbedding(nn.Module):
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
        embbeding = torch.tile(self.sorted_eigenvecs, (batch, 1, 1))
        device = embbeding.device
        x = x.to(device)
        x = x + embbeding
        return self.dropout(x)


class SklearnSpectralEmbedding(nn.Module):
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
 
        # Compute eigenvectors at init — do NOT store G or A as attributes
        G = nx.grid_2d_graph(grid_size, grid_size)
        A = nx.to_numpy_array(G)
        transformation = SpectralEmbedding(
            n_components=n_components, affinity="precomputed", **kwargs
        )
        sorted_eigenvecs = transformation.fit_transform(A)
        sorted_eigenvecs = torch.tensor(sorted_eigenvecs, dtype=torch.float32)
 
        # Register as buffer — moves with .to(device), not a parameter
        self.register_buffer("sorted_eigenvecs", sorted_eigenvecs)
 
        self.proj = nn.Linear(n_components, d_model)
        # elementwise_affine=False: proj learns direction, not scale — prevents unbounded growth
        self.proj_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.to_project = d_model != n_components
        # Normalize content to unit scale before adding SE — prevents SE (std~1.1) from
        # drowning out learned content features (std~0.15) during early training.
        self.content_norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _ = x.shape
        embedding = self.sorted_eigenvecs.unsqueeze(0).expand(batch, -1, -1)  # no copy
        if self.to_project:
            embedding = self.proj_norm(self.proj(embedding))

        # dropout before adding, not after — keeps x scale stable
        embedding = self.dropout(embedding)
        return self.content_norm(x) + embedding
