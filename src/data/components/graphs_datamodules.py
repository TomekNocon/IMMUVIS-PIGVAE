from __future__ import annotations
import torch
from typing import Optional, Callable, Union, Tuple, List
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import networkx as nx


class PatchAugmentations(nn.Module):
    """
    Apply 8 dihedral (rotation + horizontal flip) augmentations on a patch
    represented as a flattened grid of node indices.

    During validation:
        Returns all 8 augmented variants stacked in order.
    During training:
        Returns the 8 variants in a consistently permuted order.
    """

    NUM_PERM = 8  # 4 rotations x {no flip, flip}

    def __init__(
        self, prob: float, size: int, patch_size: int, is_validation: bool = False
    ):
        super().__init__()
        self.prob = prob
        self.is_validation = is_validation
        num_nodes_per_dim = size // patch_size
        self.register_buffer(
            "grid", self.make_grid(num_nodes_per_dim), persistent=False
        )

    def forward(
        self, patch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patch: Tensor of shape [C, N, D], N = patch_size^2.

        Returns:
            aug_tensor: [8, C, N, D]
            argsort_tensor: [8, N]
            perm: [8] permutation used
        """
        device = patch.device
        grid = self.grid.to(device)

        aug_list = []
        argsort_list = []

        for k in range(4):
            rotated_grid = torch.rot90(grid, k=k, dims=[0, 1])
            flat_idx = rotated_grid.flatten()
            aug_list.append(patch[:, flat_idx, :].contiguous())
            argsort_list.append(torch.argsort(flat_idx))

            flipped_grid = TF.hflip(rotated_grid)
            flat_flipped_idx = flipped_grid.flatten()
            aug_list.append(patch[:, flat_flipped_idx, :].contiguous())
            argsort_list.append(torch.argsort(flat_flipped_idx))

        aug_tensor = torch.stack(aug_list, dim=0).contiguous()  # [8, C, N, D]
        argsort_tensor = torch.stack(argsort_list, dim=0).contiguous()  # [8, N]

        if self.is_validation:
            perm = torch.arange(self.NUM_PERM, device=device)
            return aug_tensor, argsort_tensor, perm

        perm = torch.randperm(self.NUM_PERM, device=device)
        # perm = torch.arange(self.NUM_PERM, device=device)
        aug_tensor_shuffled = aug_tensor
        argsort_tensor_shuffled = argsort_tensor
        return aug_tensor_shuffled, argsort_tensor_shuffled, perm

    @staticmethod
    def make_grid(num_nodes_per_dim: int) -> torch.Tensor:
        """Create a 2D grid mapping flattened indices to 2D for rotation/flip operations."""
        return torch.arange(num_nodes_per_dim**2).reshape(
            num_nodes_per_dim, num_nodes_per_dim
        )


class DualOutputTransform:
    """
    A wrapper that returns both original and augmented versions of the image
    """

    def __init__(
        self,
        base_transforms: Union[T.Compose, Callable],
        augmentation_transforms: Union[T.Compose, Callable],
    ):
        self.base_transforms = base_transforms
        self.augmentation_transforms = augmentation_transforms

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor]:
        # Apply base transformations to get the original version
        original = self.base_transforms(img)

        # Apply the same base transformations + augmentations to get the augmented version
        augmented, argsort_augmented, perm = self.augmentation_transforms(original)
        return (augmented, argsort_augmented, perm)


class SplitPatches(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> B c h w
        # bs, c, h, w = x.shape
        bs, c, h, w = x.shape

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
        channels: List[int],
    ):
        self.grid_size = grid_size
        self.dataset = dataset
        self.channels = channels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple:
        g = nx.grid_graph((self.grid_size, self.grid_size))
        # img = self.dataset[idx][0].to(torch.float32)
        augmented, argsort_augmented, perm = self.dataset[idx][0]
        augmented = augmented.to(torch.float32)
        target = self.dataset[idx][1]
        return (g, augmented, argsort_augmented, perm, target)


class DenseGraphBatch:
    def __init__(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        argsort_augmented_features: Optional[torch.Tensor] = None,
        perms: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.node_features = node_features
        self.edge_features = edge_features
        self.mask = mask
        self.argsort_augmented_features = argsort_augmented_features
        self.perms = perms
        self.properties = kwargs.get("properties", None)

    @classmethod
    def from_sparse_graph_list(
        cls, data_list: List[Tuple], labels: bool = True
    ) -> DenseGraphBatch:
        if labels:
            max_num_nodes = max(
                [graph.number_of_nodes() for graph, _, _, _, _ in data_list]
            )
        else:
            max_num_nodes = max(
                [graph.number_of_nodes() for graph, _, _, _, _ in data_list]
            )
        node_features = []
        edge_features = []
        argsort_augmented_indices = []
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
        ) in data_list:
            y.append(label)
            num_nodes = graph.number_of_nodes()
            props.append(torch.Tensor([num_nodes]))
            graph.add_nodes_from([i for i in range(num_nodes, max_num_nodes)])
            node_features.append(augmented_embedding[perm].squeeze(1))
            argsort_augmented_indices.append(argsort_augmented[perm].squeeze(1))
            perms.append(perm.squeeze(0))
            mask.append((torch.arange(max_num_nodes) < num_nodes).unsqueeze(0))
        node_features = torch.stack(node_features, dim=1).flatten(0, 1)
        argsort_augmented_indices = torch.stack(
            argsort_augmented_indices, dim=1
        ).flatten(0, 1)
        perms = torch.stack(perms, dim=1).flatten(0, 1)
        batch_size = node_features.size(0)
        edge_features = torch.tensor(edge_features)
        mask = torch.cat(mask, dim=0)
        batch_size_mask = mask.size(0)
        factor = int(batch_size / batch_size_mask)
        mask = mask.repeat_interleave(factor, dim=0)
        props = torch.cat(props, dim=0)
        batch = DenseGraphBatch(
            node_features=node_features,
            edge_features=edge_features,
            argsort_augmented_features=argsort_augmented_indices,
            perms=perms,
            mask=mask,
            properties=props,
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

        return DenseGraphBatch(
            node_features=node_features[:n, :, :],
            edge_features=edge_features[:n],
            mask=mask[:n, :],
            argsort_augmented_features=argsort_augmented_features[:n, :, :]
            if argsort_augmented_features is not None
            else None,
            perms=perms[:n, :],
            properties=properties[:n],
        )


def dense_graph_collate_fn(data_list: List[Tuple]) -> DenseGraphBatch:
    return DenseGraphBatch.from_sparse_graph_list(data_list)


class DenseGraphDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        labels: bool = True,
        **kwargs,
    ):
        self.labels = labels
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dense_graph_collate_fn,  # Directly pass the standalone function
            **kwargs,
        )
