from __future__ import annotations
import torch
from typing import Optional, Callable, Union, Tuple, List
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import networkx as nx


# class PatchAugmentations(nn.Module):
#     """Apply one of 8 possible dihedral (rotation + flip) augmentations"""

#     NUM_PERM = 8

#     def __init__(self, prob: float, size: int, patch_size: int, is_validation: bool = False):
#         super().__init__()
#         self.prob = prob
#         self.is_validation = is_validation
#         self.grid = self.make_grid(size // patch_size)

#     def forward(self, patch: torch.Tensor) -> torch.Tensor:
#         aug_list = []
#         self.grid = self.grid.to(img.device)
#         idx  = TF.hflip(self.grid).flatten()
#         aug_list.append(img[:, idx, :])
#         for k in range(1, 4):  # rotations: 0, 90, 180, 270 degrees
#             rotated_idx = torch.rot90(self.grid, k=k, dims=[-2, -1])
#             aug_list.append(img[:, rotated_idx, :])  # no flip
#             flip_idx = TF.hflip(rotated_idx)
#             aug_list.append(img[:, flip_idx, :])  # with horizontal flip
#         aug_list = torch.stack(aug_list, dim=0).squeeze(1)
#         if self.is_validation:
#             return aug_list
#         return aug_list[
#             torch.randperm(self.NUM_PERM - 1)
#         ]  # shuffle during traininghape: [7, C, H, W]


#     def make_grid(self, num_nodes_columns: int) -> torch.Tensor:
#         return (
#             torch
#             .arange(num_nodes_columns * num_nodes_columns)
#             .reshape(num_nodes_columns, num_nodes_columns)
#         )
class PatchAugmentations(nn.Module):
    """
    Apply one of 8 possible dihedral (rotation + horizontal flip) augmentations
    on a square patch represented as a flattened grid of node indices.

    During validation:
        Returns all 8 augmented variants stacked.
    During training:
        Returns the 8 augmented variants shuffled.
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

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: Tensor of shape [C, N, D] where N = patch_size^2, or a similar layout.

        Returns:
            Tensor of shape:
                [7, C, N, D] during validation
                [7, C, N, D] shuffled during training
        """
        device = patch.device
        grid = self.grid.to(device)

        aug_list = []
        argsort_aug_list = []
        flat_flipped_idx = TF.hflip(grid).flatten()
        aug_list.append(patch[:, flat_flipped_idx, :])
        argsort_aug_list.append(torch.argsort(flat_flipped_idx))
        for k in range(1, 4):  # 0, 90, 180, 270 degrees
            rotated_grid = torch.rot90(grid, k=k, dims=[0, 1])
            flat_rotated_idx = rotated_grid.flatten()
            aug_list.append(patch[:, flat_rotated_idx, :])  # without flip

            flipped_grid = TF.hflip(rotated_grid)
            flat_flipped_idx = flipped_grid.flatten()
            aug_list.append(patch[:, flat_flipped_idx, :]) # with horizontal flip
            argsort_aug_list.append(torch.argsort(flat_rotated_idx)) 
            argsort_aug_list.append(torch.argsort(flat_flipped_idx))
        aug_tensor = torch.stack(aug_list, dim=0)  # [7, C, N, D]
        argsort_tensor = torch.stack(argsort_aug_list, dim=0) 

        if self.is_validation:
            return aug_tensor, argsort_tensor

        # Shuffle augmentations during training
        perm = torch.randperm(self.NUM_PERM - 1, device=device)
        return aug_tensor[perm], argsort_tensor[perm]

    @staticmethod
    def make_grid(num_nodes_per_dim: int) -> torch.Tensor:
        """
        Creates a 2D grid of indices with shape [num_nodes_per_dim, num_nodes_per_dim],
        mapping a flattened patch to its 2D structure.
        """
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
        patch_transform: Optional[Union[T.Compose, Callable]] = None,
    ):
        self.base_transforms = base_transforms
        self.augmentation_transforms = augmentation_transforms
        self.patch_transform = patch_transform

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor]:
        # Apply base transformations to get the original version
        original = self.base_transforms(img)

        # Apply the same base transformations + augmentations to get the augmented version
        augmented = self.augmentation_transforms(img)

        # Apply patch transform if specified
        if self.patch_transform:
            original = self.patch_transform(original)
            augmented = self.patch_transform(augmented)

        return (original, augmented)


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
        original, augmented = self.dataset[idx][0]
        original, augmented = original.to(torch.float32), augmented.to(torch.float32)
        target = self.dataset[idx][1]
        return (g, original, augmented, target)


class DenseGraphBatch:
    def __init__(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.node_features = node_features
        self.edge_features = edge_features
        self.mask = mask
        self.properties = kwargs.get("properties", None)

    @classmethod
    def from_sparse_graph_list(
        cls, data_list: List[Tuple], labels: bool = True
    ) -> DenseGraphBatch:
        if labels:
            max_num_nodes = max(
                [graph.number_of_nodes() for graph, _, _, _ in data_list]
            )
        else:
            max_num_nodes = max([graph.number_of_nodes() for graph in data_list])
        in_node_features = []
        out_node_features = []
        edge_features = []
        mask = []
        y = []
        props = []
        for graph, original_embedding, augmented_embedding, label in data_list:
            y.append(label)
            num_nodes = graph.number_of_nodes()
            props.append(torch.Tensor([num_nodes]))
            graph.add_nodes_from([i for i in range(num_nodes, max_num_nodes)])
            in_node_features.append(augmented_embedding)
            out_node_features.append(original_embedding)
            mask.append((torch.arange(max_num_nodes) < num_nodes).unsqueeze(0))
        in_node_features = torch.stack(in_node_features, dim=1)
        _, _, _, emd_dim = in_node_features.shape
        in_node_features = in_node_features.view(-1, num_nodes, emd_dim)
        out_node_features = torch.cat(out_node_features, dim=0)
        contrastive_node_features = torch.cat(
            [out_node_features, in_node_features], dim=0
        )
        batch_size = contrastive_node_features.size(0)
        edge_features = torch.tensor(edge_features)
        mask = torch.cat(mask, dim=0)
        batch_size_mask = mask.size(0)
        factor = int(batch_size / batch_size_mask)
        mask = mask.repeat_interleave(factor, dim=0)
        props = torch.cat(props, dim=0)
        batch = DenseGraphBatch(
            node_features=contrastive_node_features,
            edge_features=edge_features,
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

        return DenseGraphBatch(
            node_features=node_features[:n, :, :],
            edge_features=edge_features[:n],
            mask=mask[:n, :],
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
