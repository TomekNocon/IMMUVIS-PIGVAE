from __future__ import annotations

import math
from collections.abc import Callable
from typing import ClassVar

import h5py
import joblib
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset

IMC_GRAPH_VIEW_KEYS: tuple[str, ...] = (
    "r0_f",
    "r0_nf",
    "r180_f",
    "r180_nf",
    "r270_f",
    "r270_nf",
    "r90_f",
    "r90_nf",
)


class PCALayer(nn.Module):
    def __init__(
        self,
        pca_path,
        statistics_path,
        clip_range: float = 0.0,
        zscore: bool = True,
    ):
        super().__init__()
        pca = joblib.load(pca_path)
        statistics = torch.load(statistics_path)

        self.register_buffer("components", torch.tensor(pca.components_, dtype=torch.float32))
        self.register_buffer("pca_mean", torch.tensor(pca.mean_, dtype=torch.float32))
        self.register_buffer("mean", torch.tensor(statistics["mean"], dtype=torch.float32))
        self.register_buffer("std", torch.tensor(statistics["std"], dtype=torch.float32))
        self.clip_range = clip_range
        self.zscore = zscore

    def forward(self, x):
        """
        x shape: [Batch, Nodes, 512] (e.g., [64, 49, 512])
        """
        x = x - self.pca_mean
        x = torch.matmul(x, self.components.t())
        if self.zscore:
            x = (x - self.mean) / (self.std + 1e-8)
        if self.clip_range > 0:
            x = x.clamp(-self.clip_range, self.clip_range)
        return x

    def inverse(self, z):
        """
        z shape: [Batch, Nodes, 128]
        """
        if self.zscore:
            z = (z * self.std) + self.mean
        return torch.matmul(z, self.components) + self.pca_mean


def _spatial_view(x: np.ndarray, key: str) -> np.ndarray:
    """Apply rotation + optional flip to a (C, H, W) array."""
    rot_part, flip_part = key.split("_")
    k = int(rot_part[1:]) // 90
    out = np.rot90(x, k=k, axes=(1, 2)) if k > 0 else x
    if flip_part == "f":
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)


def _center_crop_np(x: np.ndarray, size: int) -> np.ndarray:
    """Center-crop a (C, H, W) array to (C, size, size)."""
    _, h, w = x.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return x[:, top : top + size, left : left + size]


def make_views(x: np.ndarray, center_crop_size: int | None = None) -> np.ndarray:
    """Generate 8 spatial views for a single (C, H, W) embedding.

    If *center_crop_size* is given the crop is applied **before** the
    rotations/flips so that every view contains exactly the same set of values.

    Returns an (8, C, H, W) array ordered by ``IMC_GRAPH_VIEW_KEYS``.
    """
    if center_crop_size is not None:
        x = _center_crop_np(x, center_crop_size)
    return np.stack([_spatial_view(x, k) for k in IMC_GRAPH_VIEW_KEYS], axis=0)


class PickleDataset(Dataset):
    def __init__(
        self,
        hdf5_path,
        transform=None,
        only_embeddings: bool = False,
        generate_views: bool = False,
        center_crop_size: int | None = None,
    ):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.only_embeddings = only_embeddings
        self.generate_views = generate_views
        self.center_crop_size = center_crop_size

        with h5py.File(hdf5_path, "r") as f:
            self._length = len(f[next(iter(f.keys()))])

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, "r") as f:
            item = {key: f[key][idx] for key in f.keys()}

        if self.generate_views and "embeddings" in item:
            emb = item["embeddings"]
            if emb.ndim == 3:
                item["embeddings"] = make_views(emb, self.center_crop_size)

        if self.transform:
            item = self.transform(item)

        if self.only_embeddings:
            item = item[0]
        return item


class PatchAugmentations(nn.Module):
    NUM_PERM = 8  # 4 rotations x {no flip, flip}

    def __init__(
        self,
        prob: float,
        size: int,
        patch_size: int,
        is_validation: bool = False,
        center_crop_size: int | None = None,
    ):
        super().__init__()
        self.prob = prob
        self.is_validation = is_validation
        num_nodes_per_dim = (
            size // patch_size if center_crop_size is None else center_crop_size // patch_size
        )
        self.register_buffer("grid", self.make_grid(num_nodes_per_dim), persistent=False)

    def forward(
        self, patches: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: Dict[str, Tensor], each [C, N, D]
        Returns:
            aug_tensor: [8, C, N, D]
            argsort_tensor: [8, N]
            perm: [8] permutation used
        """
        device = self.grid.device
        grid = self.grid

        aug_list, argsort_list = [], []

        for transform_key, patch_embedding in patches.items():
            if transform_key == "img_path":
                continue

            # transformed_grid = self.apply_transform(grid, transform_key)
            transformed_grid = grid
            flat_idx = transformed_grid.flatten()

            aug_list.append(patch_embedding)  # or patch_embedding if already Tensor
            argsort_list.append(torch.argsort(flat_idx))

        aug_tensor = torch.stack(aug_list, dim=0).contiguous()
        argsort_tensor = torch.stack(argsort_list, dim=0).contiguous()

        if self.is_validation:
            perm = torch.arange(self.NUM_PERM, device=device)
            return aug_tensor, argsort_tensor, perm

        # perm = torch.randperm(self.NUM_PERM, device=device)
        perm = torch.arange(self.NUM_PERM, device=device)
        return aug_tensor, argsort_tensor, perm

    @staticmethod
    def make_grid(num_nodes_per_dim: int) -> torch.Tensor:
        """Create a 2D grid mapping flattened indices to 2D for rotation/flip
        operations."""
        return torch.arange(num_nodes_per_dim**2).reshape(num_nodes_per_dim, num_nodes_per_dim)

    @staticmethod
    def apply_transform(grid: torch.Tensor, key: str) -> torch.Tensor:
        """Apply rotation/flip based on key string (e.g. 'r90_f')."""
        # Parse
        rot_part, flip_part = key.split("_")
        angle = int(rot_part[1:])  # 'r90' -> 90
        flip = flip_part == "f"

        # Apply rotation
        k = angle // 90
        out = torch.rot90(grid, k=k, dims=[0, 1]) if k > 0 else grid

        # Apply flip
        if flip:
            out = torch.flip(out, dims=[-1])

        return out


class IMCBaseDictTransform(nn.Module):
    keys: ClassVar[tuple[str, ...]] = IMC_GRAPH_VIEW_KEYS
    def __init__(
        self,
        exclude_metadata: list[str] | None = None,
        center_crop_size: int | None = None,
        normalize: bool = False,
        norm_type: str = "channel_wise",  # "channel_wise", "global", or "none"
        clip_percentiles: bool = False,
        clip_lower: float = 0.01,
        clip_upper: float = 0.99,
        clip_type: str = "channel_wise",  # "channel_wise", "global", or "none"
    ):
        """Transform for IMC embeddings with proper normalization.

        Args:
            exclude_metadata: Keys to exclude from processing
            apply_center_crop: Whether to apply center crop
            normalize: Whether to normalize features
            norm_type: Type of normalization:
                - "channel_wise": Normalize each channel independently (recommended)
                - "global": Normalize all features together
                - "none": No normalization
            clip_percentiles: Whether to clip values to specified percentiles
            clip_lower: Lower percentile (e.g., 0.01 for 1st percentile)
            clip_upper: Upper percentile (e.g., 0.99 for 99th percentile)
            clip_type: Percentile clipping mode: "channel_wise", "global", or "none"
        """
        super().__init__()
        self.exclude_metadata = exclude_metadata
        self.center_crop_size = center_crop_size
        self.normalize = normalize
        self.norm_type = norm_type
        self.clip_percentiles = clip_percentiles
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        self.clip_type = clip_type

    def forward(
        self, embeddings: dict, mean: torch.Tensor | None = None, std: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        data: dict[str, torch.Tensor] = {}
        for key, embedding in zip(self.keys, embeddings, strict=True):
            if not isinstance(embedding, torch.Tensor) and (
                self.exclude_metadata is None or key not in self.exclude_metadata
            ):
                embedding = torch.from_numpy(embedding)
                embedding = embedding.squeeze(0)
                c, _, _ = embedding.shape

                # if self.center_crop_size:
                #     center_crop = T.CenterCrop((self.center_crop_size, self.center_crop_size))
                #     embedding = center_crop(embedding)
                #     c, _, _ = embedding.shape  # Update dimensions after crop

                if self.clip_percentiles and self.clip_type != "none":
                    embedding = self._clip_by_percentile(
                        embedding, lower=self.clip_lower, upper=self.clip_upper, mode=self.clip_type
                    )

                if self.normalize:  # and mean is not None and std is not None:
                    if self.norm_type == "channel_wise":
                        # Normalize each channel independently (preserves relative structure)
                        # Shape: [C, H, W]
                        embedding = self._normalize_channel_wise(embedding, mean=mean, std=std)
                    elif self.norm_type == "global":
                        # Normalize all features together
                        embedding = self._normalize_global(embedding)

                # embedding = torch.arcsinh(embedding / 5)
                # Reshape to [N, C] where N = H*W
                embedding = embedding.reshape(c, -1).T

            data[key] = embedding
        return data

    def _clip_by_percentile(
        self, x: torch.Tensor, lower: float, upper: float, mode: str = "channel_wise"
    ) -> torch.Tensor:
        """Clip tensor values between given lower/upper percentiles.

        Args:
            x: Tensor of shape [C, H, W]
            lower: Lower percentile in [0, 1]
            upper: Upper percentile in [0, 1]
            mode: "channel_wise" or "global"
        """
        if lower is None or upper is None or lower >= upper:
            return x
        if mode == "global":
            flat = x.flatten()
            q_low = torch.quantile(flat, lower)
            q_high = torch.quantile(flat, upper)
            return torch.clamp(x, min=q_low, max=q_high)
        elif mode == "channel_wise":
            c = x.shape[0]
            x_reshaped = x.reshape(c, -1)
            q_low = torch.quantile(x_reshaped, lower, dim=1, keepdim=True).reshape(c, 1, 1)
            q_high = torch.quantile(x_reshaped, upper, dim=1, keepdim=True).reshape(c, 1, 1)
            return torch.maximum(torch.minimum(x, q_high), q_low)
        else:
            return x

    def _normalize_channel_wise(
        self, x: torch.Tensor, mean: torch.Tensor | None = None, std: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Normalize each channel (feature dimension) independently.

        This is critical for feature maps from encoders.
        """
        # x shape: [C, H, W]
        eps = 1e-6

        if mean is None or std is None:
            mean = x.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
            std = x.std(dim=(1, 2), keepdim=True) + eps  # [C, 1, 1]
        else:
            mean = mean.unsqueeze(1).unsqueeze(2)
            std = std.unsqueeze(1).unsqueeze(2)

        # Standardize
        x_norm = (x - mean) / std

        return x_norm

    def _normalize_global(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize all features together."""
        eps = 1e-6
        mean = x.mean()
        std = x.std() + eps
        return (x - mean) / std


class DualOutputTransform:
    """A wrapper that returns both original and augmented versions of the image."""

    def __init__(
        self,
        base_transforms: T.Compose | Callable,
        augmentation_transforms: T.Compose | Callable,
    ):
        self.base_transforms = base_transforms
        self.augmentation_transforms = augmentation_transforms
        self.mean = None
        self.std = None

    def __call__(self, img: torch.Tensor) -> tuple[object, object, object, object, object, object]:
        # Apply base transformations to get the original version
        original = img
        if self.base_transforms is not None:
            original["embeddings"] = self.base_transforms(
                img["embeddings"], mean=self.mean, std=self.std
            )

        # Apply the same base transformations + augmentations to get the augmented version
        augmented, argsort_augmented, perm = self.augmentation_transforms(original["embeddings"])
        return (
            augmented,
            argsort_augmented,
            perm,
            original["metadata"],
            original["paths"],
            original["positions"],
        )

    def set_mean(self, mean: torch.Tensor):
        self.mean = mean

    def set_std(self, std: torch.Tensor):
        self.std = std


class SplitPatches(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> B c h w
        # bs, c, h, w = x.shape
        bs, c, _, _ = x.shape

        x = self.unfold(x)
        # x -> B (c*p*p) L

        # Reshaping into the shape we want
        a = x.view(bs, c, self.patch_size, self.patch_size, -1).permute(0, 4, 1, 2, 3)
        a = a.view(bs, -1, c * self.patch_size * self.patch_size)
        # a -> ( B no.of patches c p p )
        return a


class GridGraphDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        grid_size: int,
        channels: list[int],
    ):
        self.grid_size = grid_size
        self.dataset = dataset
        self.channels = channels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        augmented, argsort_augmented, perm, metadata, paths, positions = self.dataset[idx]
        augmented = augmented.to(torch.float32)
        metadata = torch.from_numpy(metadata)
        positions = torch.from_numpy(positions)
        true_grid_size = int(math.sqrt(augmented.shape[1]))
        true_grid_size = min(true_grid_size, self.grid_size)
        g = nx.grid_graph((true_grid_size, true_grid_size))
        # augmented = augmented[:, :, self.channels]
        target = -1
        return (g, augmented, argsort_augmented, perm, target, metadata, paths, positions)


class DenseGraphBatch:
    def __init__(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        argsort_augmented_features: torch.Tensor | None = None,
        perms: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
        paths: np.ndarray | None = None,
        positions: torch.Tensor | None = None,
        **kwargs,
    ):
        self.node_features = node_features
        self.edge_features = edge_features
        self.mask = mask
        self.argsort_augmented_features = argsort_augmented_features
        self.perms = perms
        self.properties = kwargs.get("properties", None)
        self.metadata = metadata
        self.paths = paths
        self.positions = positions

    def to(self, device):
        """Move all tensors in the batch to the specified device."""
        self.node_features = self.node_features.to(device)
        self.edge_features = self.edge_features.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        if self.argsort_augmented_features is not None:
            self.argsort_augmented_features = self.argsort_augmented_features.to(device)
        if self.perms is not None:
            self.perms = self.perms.to(device)
        if self.properties is not None:
            self.properties = self.properties.to(device)
        if hasattr(self, "y") and self.y is not None:
            self.y = self.y.to(device)
        if self.metadata is not None:
            self.metadata = self.metadata.to(device)
        if self.positions is not None:
            self.positions = self.positions.to(device)
        return self

    @classmethod
    def from_sparse_graph_list(cls, data_list: list[tuple], labels: bool = True) -> DenseGraphBatch:
        if labels:
            max_num_nodes = max([
                graph.number_of_nodes() for graph, _, _, _, _, _, _, _ in data_list
            ])
        else:
            max_num_nodes = max([
                graph.number_of_nodes() for graph, _, _, _, _, _, _, _ in data_list
            ])
        node_features = []
        edge_features_tensor = torch.empty(0)
        argsort_augmented_indices = []
        metadata_list = []
        paths_list = []
        positions_list = []
        mask = []
        y = []
        props = []
        perms = []
        for (
            graph,
            augmented_embedding,
            argsort_augmented,
            perm,
            label,
            metadata_item,
            paths_item,
            positions_item,
        ) in data_list:
            y.append(label)
            num_nodes = graph.number_of_nodes()
            props.append(torch.Tensor([num_nodes]))
            graph.add_nodes_from(list(range(num_nodes, max_num_nodes)))
            node_features.append(augmented_embedding[perm].squeeze(1))
            argsort_augmented_indices.append(argsort_augmented[perm].squeeze(1))
            perms.append(perm.squeeze(0))
            mask.append((torch.arange(max_num_nodes) < num_nodes).unsqueeze(0))
            metadata_list.append(metadata_item)
            paths_list.append(paths_item)
            positions_list.append(positions_item)
        node_features = torch.stack(node_features, dim=1).flatten(0, 1)
        argsort_augmented_indices = torch.stack(argsort_augmented_indices, dim=1).flatten(0, 1)
        perms = torch.stack(perms, dim=1).flatten(0, 1)
        batch_size = node_features.size(0)
        edge_features = edge_features_tensor
        mask = torch.cat(mask, dim=0)
        batch_size_mask = mask.size(0)
        factor = int(batch_size / batch_size_mask)
        mask = mask.repeat_interleave(factor, dim=0)
        props = torch.cat(props, dim=0)
        metadata = torch.cat(metadata_list, dim=0)
        # Keep paths as numpy array (could be strings or non-tensor types)
        try:
            paths = np.stack(paths_list, axis=0)
        except Exception:
            paths = np.array(paths_list)
        positions = torch.cat(positions_list, dim=0)
        batch = DenseGraphBatch(
            node_features=node_features,
            edge_features=edge_features,
            argsort_augmented_features=argsort_augmented_indices,
            perms=perms,
            mask=mask,
            properties=props,
            metadata=metadata,
            paths=paths,
            positions=positions,
        )
        if labels:
            batch.y = torch.Tensor(y)
        return batch

    def take_sample(self, n) -> DenseGraphBatch:
        node_features = self.node_features
        edge_features = self.edge_features
        mask = self.mask
        properties = self.properties
        argsort_augmented_features = self.argsort_augmented_features
        perms = self.perms
        metadata = self.metadata
        paths = self.paths
        positions = self.positions
        return DenseGraphBatch(
            node_features=node_features[:n, :, :],
            edge_features=edge_features[:n],
            mask=mask[:n, :] if mask is not None else None,
            argsort_augmented_features=argsort_augmented_features[:n, :, :]
            if argsort_augmented_features is not None
            else None,
            perms=perms[:n, :] if perms is not None else None,
            properties=properties[:n] if properties is not None else None,
            metadata=metadata[:n, :] if metadata is not None else None,
            paths=(
                paths[:n]
                if paths is not None and isinstance(paths, np.ndarray) and paths.ndim == 1
                else (paths[:n, :] if paths is not None else None)
            ),
            positions=positions[:n, :] if positions is not None else None,
        )


def dense_graph_collate_fn(data_list: list[tuple]) -> DenseGraphBatch:
    return DenseGraphBatch.from_sparse_graph_list(data_list)


class PCADenseGraphCollator:
    """Apply PCA to batched node features during collation.

    If `keep_input` is set, the raw pre-PCA node features are stashed on the batch as
    `input_features` (for offline inverse-PCA-vs-input diagnostics). Off by default so
    training is unaffected.
    """

    def __init__(self, pca_layer: PCALayer, keep_input: bool = False):
        self.pca_layer = pca_layer
        self.keep_input = keep_input

    def __call__(self, data_list: list[tuple]) -> DenseGraphBatch:
        batch = DenseGraphBatch.from_sparse_graph_list(data_list)
        with torch.no_grad():
            if self.keep_input:
                batch.input_features = batch.node_features  # raw, pre-PCA (D_orig)
            batch.node_features = self.pca_layer(batch.node_features)
        return batch


class DenseGraphDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        labels: bool = True,
        collate_fn: Callable | None = None,
        **kwargs,
    ):
        self.labels = labels
        if collate_fn is None:
            collate_fn = dense_graph_collate_fn
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,  # Directly pass the standalone function
            **kwargs,
        )


class WelfordOnline:
    def __init__(self, channels: int):
        self.count = 0
        # Internal accumulators in float64 for precision
        self.mean = torch.zeros(channels, dtype=torch.float64)
        self.m2 = torch.zeros(channels, dtype=torch.float64)

    def update(self, x: torch.Tensor):
        """
        x: (N, channels) where N is (batch_size * seq_len)
        """
        x = x.to(torch.float64)
        batch_n = x.size(0)
        
        # Batch-wise statistics
        batch_mean = x.mean(dim=0)
        # Sum of squares of deviations from the batch mean
        batch_m2 = ((x - batch_mean) ** 2).sum(dim=0)

        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.count = batch_n
        else:
            total_n = self.count + batch_n
            delta = batch_mean - self.mean
            
            # Welford/Chan update formula for merging two sets
            self.mean += delta * (batch_n / total_n)
            self.m2 += batch_m2 + (delta**2) * (self.count * batch_n / total_n)
            
            self.count = total_n

    def finalize(self):
        if self.count < 1:
            return self.mean.float(), torch.ones_like(self.mean).float()
        
        variance = self.m2 / self.count
        std = torch.sqrt(torch.maximum(variance, torch.tensor(1e-8)))
        
        return self.mean.float(), std.float()
