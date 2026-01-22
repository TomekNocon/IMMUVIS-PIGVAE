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
        self.G = nx.grid_2d_graph(grid_size, grid_size)
        self.A = nx.to_numpy_array(self.G)
        transformation = SpectralEmbedding(
            n_components=n_components, affinity="precomputed", **kwargs
        )
        sorted_eigenvecs = transformation.fit_transform(self.A)
        sorted_eigenvecs = torch.tensor(sorted_eigenvecs, dtype=torch.float32)

        self.register_buffer("sorted_eigenvecs", sorted_eigenvecs)
        self.proj = nn.Linear(n_components, d_model)
        self.to_project = d_model != n_components
        # Normalize & scale the (fixed) spectral positional signal before adding.
        # This helps prevent the spectral term from dominating early in training.
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_scale = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _ = x.shape
        embedding = torch.tile(self.sorted_eigenvecs, (batch, 1, 1))
        if self.to_project:
            embedding = self.proj(embedding)
        embedding = self.emb_norm(embedding)
        x = x + self.emb_scale.to(dtype=x.dtype, device=x.device) * embedding.to(dtype=x.dtype)
        return self.dropout(x)
