# tests/test_single_view_transform.py
import numpy as np
import pytest
import torch

from src.data.components.graphs_datamodules import (
    DenseGraphBatch,
    SingleViewTransform,
)


def make_fake_emb(C=4, H=6, W=6):
    return np.random.randn(C, H, W).astype(np.float32)


class TestSingleViewTransform:
    def test_output_shapes(self):
        t = SingleViewTransform(is_validation=True)
        emb = make_fake_emb(C=4, H=6, W=6)
        aug, argsort, perm = t(emb)
        assert aug.shape == (1, 36, 4), f"aug shape wrong: {aug.shape}"
        assert argsort.shape == (1, 36), f"argsort shape wrong: {argsort.shape}"
        assert perm.shape == (1,), f"perm shape wrong: {perm.shape}"
        assert perm[0] == 0

    def test_validation_is_deterministic(self):
        t = SingleViewTransform(is_validation=True)
        emb = make_fake_emb()
        aug1, _, _ = t(emb)
        aug2, _, _ = t(emb)
        assert torch.allclose(aug1, aug2)

    def test_training_returns_tensor(self):
        t = SingleViewTransform(is_validation=False)
        emb = make_fake_emb()
        aug, argsort, perm = t(emb)
        assert aug.dtype == torch.float32
        assert argsort.dtype == torch.int64

    def test_center_crop(self):
        t = SingleViewTransform(is_validation=True, center_crop_size=4)
        emb = make_fake_emb(C=4, H=6, W=6)
        aug, argsort, perm = t(emb)
        assert aug.shape == (1, 16, 4)  # 4x4 = 16 nodes

    def test_perms_stack_in_collator(self):
        """Regression: perm.squeeze(0) was scalar for shape [1], crashing stack."""
        import networkx as nx

        t = SingleViewTransform(is_validation=True)
        emb = make_fake_emb(C=4, H=6, W=6)
        aug, argsort, perm = t(emb)  # [1, 36, 4], [1, 36], [1]

        g = nx.grid_graph((6, 6))
        data_list = [
            (g, aug, argsort, perm, -1,
             torch.zeros(1, dtype=torch.long),
             np.array(["path"]),
             torch.zeros(1, 2))
            for _ in range(2)
        ]
        batch = DenseGraphBatch.from_sparse_graph_list(data_list)
        assert batch.node_features.shape == (2, 36, 4)
        assert batch.perms.shape == (2, 1)
