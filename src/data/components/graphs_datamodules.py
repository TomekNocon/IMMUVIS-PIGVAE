import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
import networkx as nx

class ImageAugmentations(nn.Module):
    """Class to handle image augmentations"""
    def __init__(self, prob=1.0):
        super().__init__()
        self.prob = prob
    
    @staticmethod
    def rotate_90(img):
        """Rotate image by 90 degrees clockwise"""
        return torch.rot90(img, k=1, dims=[-2, -1])
    
    @staticmethod
    def rotate_180(img):
        """Rotate image by 180 degrees"""
        return torch.rot90(img, k=2, dims=[-2, -1])
    
    @staticmethod
    def rotate_270(img):
        """Rotate image by 270 degrees clockwise (or 90 counter-clockwise)"""
        return torch.rot90(img, k=3, dims=[-2, -1])
    
    @staticmethod
    def horizontal_flip(img):
        """Flip image horizontally"""
        return TF.hflip(img)
    
    @staticmethod
    def vertical_flip(img):
        """Flip image vertically"""
        return TF.vflip(img)

    def forward(self, img):
        """Apply a random augmentation with a certain probability"""
        if random.random() < self.prob:
            aug_type = random.choice(['rot90', 'rot180', 'rot270', 'hflip', 'vflip'])
            
            if aug_type == 'rot90':
                return ImageAugmentations.rotate_90(img)
            elif aug_type == 'rot180':
                return ImageAugmentations.rotate_180(img)
            elif aug_type == 'rot270':
                return ImageAugmentations.rotate_270(img)
            elif aug_type == 'hflip':
                return ImageAugmentations.horizontal_flip(img)
            elif aug_type == 'vflip':
                return ImageAugmentations.vertical_flip(img)
        
        return img
    
class DualOutputTransform:
    """
    A wrapper that returns both original and augmented versions of the image
    """
    def __init__(self, base_transforms, augmentation_transforms, patch_transform=None):
        self.base_transforms = base_transforms
        self.augmentation_transforms = augmentation_transforms
        self.patch_transform = patch_transform
        
    def __call__(self, img):
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
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
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
        dataset,
        grid_size: int,
        channels: list,
    ):
        self.grid_size = grid_size
        self.dataset = dataset
        self.channels = channels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        g = nx.grid_graph((self.grid_size, self.grid_size))
        # img = self.dataset[idx][0].to(torch.float32)
        original, augmented = self.dataset[idx][0]
        original, augmented  = original.to(torch.float32), augmented.to(torch.float32)
        target = self.dataset[idx][1]
        return (g, original, augmented, target)


class DenseGraphBatch:
    def __init__(self, node_features, edge_features, mask, **kwargs):
        self.node_features = node_features
        self.edge_features = edge_features
        self.mask = mask
        self.properties = kwargs.get("properties", None)

    @classmethod
    def from_sparse_graph_list(cls, data_list, labels=True):
        if labels:
            max_num_nodes = max([graph.number_of_nodes() for graph, _, _, _ in data_list])
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
        in_node_features = torch.cat(in_node_features, dim=0)
        out_node_features = torch.cat(out_node_features, dim=0)
        contrastive_node_features = torch.cat([out_node_features, in_node_features], dim=0)
        edge_features = torch.tensor(edge_features)
        mask = torch.cat(mask, dim=0)
        mask = torch.cat([mask, mask], dim=0)
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


def dense_graph_collate_fn(data_list):
    return DenseGraphBatch.from_sparse_graph_list(data_list)

class DenseGraphDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, labels=True, **kwargs):
        self.labels = labels
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dense_graph_collate_fn,  # Directly pass the standalone function
            **kwargs,
        )

