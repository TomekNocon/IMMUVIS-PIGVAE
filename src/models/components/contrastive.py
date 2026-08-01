import torch

from src.data.components.graphs_datamodules import DenseGraphBatch


def drop_views(
    batch: DenseGraphBatch,
    p: float = 0.20,
    n: int = 2,
    generator: torch.Generator | None = None,
) -> list[DenseGraphBatch]:
    """Return `n` node-dropout views of `batch`.

    Each view masks out a fresh random `p`-fraction of the currently-valid nodes
    (mask AND ~drop), never re-activating padded nodes, guaranteeing >=1 valid node
    per row. `node_features`/`edge_features` are shared (unchanged) — only `mask`
    differs, so this is a cheap on-GPU augmentation. Pure function (no model state).
    """
    B, N = batch.mask.shape
    device = batch.mask.device
    first_valid = batch.mask.float().argmax(dim=1)  # first True index per row
    views: list[DenseGraphBatch] = []
    for _ in range(n):
        rand = torch.rand(B, N, device=device, generator=generator)
        new_mask = batch.mask & (rand >= p)
        empty = ~new_mask.any(dim=1)
        if empty.any():
            new_mask[empty, first_valid[empty]] = True
        views.append(
            DenseGraphBatch(
                node_features=batch.node_features,
                edge_features=batch.edge_features,
                mask=new_mask,
            )
        )
    return views
