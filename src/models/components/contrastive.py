import torch
import torch.nn as nn

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

    When `generator` is provided, random values are drawn on the generator's device
    and moved to the batch's device to handle device mismatches (e.g., CPU generator
    with CUDA batch). When `generator is None`, random values are drawn directly on
    the batch's device.
    """
    B, N = batch.mask.shape
    device = batch.mask.device
    first_valid = batch.mask.float().argmax(dim=1)  # first True index per row
    views: list[DenseGraphBatch] = []
    for _ in range(n):
        if generator is not None:
            # Draw on generator's device, then move to batch device for safety
            gen_device = torch.device(generator.device)
            rand = torch.rand(B, N, device=gen_device, generator=generator).to(device)
        else:
            # Draw directly on batch device when no generator
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


def block_drop(
    batch: DenseGraphBatch,
    frac: float = 0.5,
    grid_size: int | None = None,
    generator: torch.Generator | None = None,
) -> DenseGraphBatch:
    """Return one view with a contiguous ~`frac` block of grid nodes removed.

    A random-position rectangular block covering ≈`frac` of the `grid_size`×`grid_size`
    grid has its `node_features` **zeroed** (so the encoder carries no content there and
    the per-node latent cannot leak the answer) and its `mask` set False (`mask AND
    ~block`). Node order is row-major (`idx = row*grid + col`). `node_features` is
    cloned (the clean tensor is the recon target — never mutate it); `edge_features`
    is shared. Guarantees ≥1 kept node per row. Device/seed-deterministic.
    """
    B, N = batch.mask.shape
    device = batch.mask.device
    grid = int(round(N ** 0.5)) if grid_size is None else int(grid_size)

    # Fixed block size derived from frac (random position), avoids degenerate slivers.
    h = max(1, min(grid, round(grid * (frac ** 0.5))))
    w = max(1, min(grid, -(-int(round(frac * N)) // h)))  # ceil(frac*N / h), clamped
    w = min(w, grid)

    def _randint(high: int) -> torch.Tensor:
        # high is exclusive upper bound for the top-left coord; high>=1.
        if generator is not None:
            gen_device = torch.device(generator.device)
            r = torch.randint(0, high, (B,), device=gen_device, generator=generator)
            return r.to(device)
        return torch.randint(0, high, (B,), device=device)

    top = _randint(grid - h + 1)                 # [B]
    left = _randint(grid - w + 1)                # [B]

    rows = torch.arange(grid, device=device).view(1, grid, 1)   # [1, grid, 1]
    cols = torch.arange(grid, device=device).view(1, 1, grid)   # [1, 1, grid]
    row_in = (rows >= top.view(B, 1, 1)) & (rows < (top + h).view(B, 1, 1))
    col_in = (cols >= left.view(B, 1, 1)) & (cols < (left + w).view(B, 1, 1))
    block = (row_in & col_in).reshape(B, N)      # [B, N] True inside block

    new_mask = batch.mask & ~block
    empty = ~new_mask.any(dim=1)
    first_valid = None  # computed below if needed
    if empty.any():
        first_valid = batch.mask.float().argmax(dim=1)
        new_mask[empty, first_valid[empty]] = True

    node_features = batch.node_features.clone()
    node_features[block] = 0.0

    # Restore features for nodes that were restored by the empty-row fallback.
    # A restored node must have non-zero features so it is consistent with mask=True.
    if empty.any():
        for b in range(B):
            if empty[b]:
                n = first_valid[b].item()
                node_features[b, n] = batch.node_features[b, n]

    return DenseGraphBatch(
        node_features=node_features,
        edge_features=batch.edge_features,
        mask=new_mask,
    )


class ProjectionHead(nn.Module):
    """2-layer MLP applied to z_global for the contrastive loss only.

    Discarded at inference — downstream reads raw z_global, so the encode/probe
    pipeline is unaffected.
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
