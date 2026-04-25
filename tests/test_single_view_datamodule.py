# tests/test_single_view_datamodule.py
import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.components.graphs_datamodules import DualOutputTransform, SingleViewTransform


class TestSingleViewDataModuleWiring:
    def test_dual_output_transform_with_none_base(self):
        """DualOutputTransform with base_transforms=None passes raw emb to SingleViewTransform."""
        svt = SingleViewTransform(is_validation=True, center_crop_size=6)
        dual = DualOutputTransform(base_transforms=None, augmentation_transforms=svt)
        emb = np.random.randn(4, 6, 6).astype(np.float32)
        fake_item = {
            "embeddings": emb,
            "metadata": torch.zeros(1, dtype=torch.long),
            "paths": np.array(["path"]),
            "positions": torch.zeros(1, 2),
        }
        aug, argsort, perm, meta, paths, pos = dual(fake_item)
        assert aug.shape == (1, 36, 4)
        assert perm[0] == 0
