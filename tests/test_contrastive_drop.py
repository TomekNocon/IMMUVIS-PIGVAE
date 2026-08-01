import torch
from src.data.components.graphs_datamodules import DenseGraphBatch


def _full_batch(B=4, N=256, D=8):
    return DenseGraphBatch(
        node_features=torch.randn(B, N, D),
        edge_features=torch.empty(0),
        mask=torch.ones(B, N, dtype=torch.bool),
    )


def test_drop_views_masks_fraction_shares_features_and_differs():
    from src.models.components.contrastive import drop_views
    batch = _full_batch()
    g = torch.Generator().manual_seed(0)
    a, b = drop_views(batch, p=0.2, n=2, generator=g)
    # ~20% dropped -> ~80% kept
    assert 0.70 < a.mask.float().mean().item() < 0.90
    # node_features untouched and shared (same object)
    assert torch.equal(a.node_features, batch.node_features)
    # the two views differ
    assert not torch.equal(a.mask, b.mask)
    # >=1 valid node per row
    assert a.mask.any(dim=1).all() and b.mask.any(dim=1).all()


def test_drop_views_never_reactivates_padded_nodes():
    from src.models.components.contrastive import drop_views
    batch = _full_batch()
    batch.mask[:, 200:] = False          # simulate padding
    a, = drop_views(batch, p=0.5, n=1, generator=torch.Generator().manual_seed(1))
    # dropped mask is a subset of the original valid nodes
    assert (a.mask & ~batch.mask).sum() == 0


def test_drop_views_reproducible_with_same_seed():
    from src.models.components.contrastive import drop_views
    batch = _full_batch()
    # First run with seeded generator
    g1 = torch.Generator().manual_seed(42)
    views1 = drop_views(batch, p=0.2, n=2, generator=g1)
    # Second run with same seed
    g2 = torch.Generator().manual_seed(42)
    views2 = drop_views(batch, p=0.2, n=2, generator=g2)
    # Masks should be identical with same seed
    assert torch.equal(views1[0].mask, views2[0].mask)
    assert torch.equal(views1[1].mask, views2[1].mask)


def test_projection_head_shape_and_grad():
    import torch
    from src.models.components.contrastive import ProjectionHead
    head = ProjectionHead(in_dim=512, hidden_dim=512, out_dim=128)
    x = torch.randn(6, 512, requires_grad=True)
    y = head(x)
    assert y.shape == (6, 128)
    y.sum().backward()
    assert x.grad is not None


def _zg_dim(gae, batch):
    with torch.no_grad():
        _, zg, *_ = gae.encode(batch, sample=False)
    return zg.shape[-1]


def test_dropped_zglobal_differs_but_identical_inputs_match():
    # The core insight, as a test: identical inputs -> identical z_global (permutation
    # views are degenerate); dropped inputs -> genuinely different z_global.
    from src.downstream.encode import _build_pl_module
    from src.models.components.contrastive import drop_views
    pl = _build_pl_module("vae16_fb0p0_film")
    gae = pl.graph_ae.eval()
    batch = DenseGraphBatch(
        node_features=torch.randn(3, 256, 128),
        edge_features=torch.empty(0),
        mask=torch.ones(3, 256, dtype=torch.bool),
    )
    with torch.no_grad():
        _, zg1, *_ = gae.encode(batch, sample=False)
        _, zg2, *_ = gae.encode(batch, sample=False)
        assert torch.allclose(zg1, zg2, atol=1e-5)          # deterministic & identical
        a, b = drop_views(batch, p=0.2, n=2, generator=torch.Generator().manual_seed(0))
        _, zga, *_ = gae.encode(a, sample=False)
        _, zgb, *_ = gae.encode(b, sample=False)
        assert not torch.allclose(zga, zgb, atol=1e-3)      # genuinely different


def test_node_stats_masked_is_noop_when_all_valid_and_ignores_dropped():
    import torch
    from src.models.components.modules import NodeStatsProjection
    head = NodeStatsProjection(hidden_dim=8)
    torch.nn.init.normal_(head.proj.weight)          # un-zero the zero-init so stats are observable
    x = torch.randn(2, 10, 8)
    full = torch.ones(2, 10, dtype=torch.bool)
    # all-valid mask == no mask (no-op guarantee for baselines)
    assert torch.allclose(head(x, full), head(x, None), atol=1e-6)
    # masked stats ignore the VALUES of dropped nodes
    m = full.clone(); m[:, 5:] = False
    x_garbage = x.clone(); x_garbage[:, 5:] = 999.0
    assert torch.allclose(head(x, m), head(x_garbage, m), atol=1e-6)


def test_encoder_zglobal_ignores_masked_node_values():
    # The whole-encoder invariant: z_global must not depend on the feature values
    # of masked-out nodes (attention + stats both honor the mask).
    import torch
    from src.downstream.encode import _build_pl_module
    from src.data.components.graphs_datamodules import DenseGraphBatch
    gae = _build_pl_module("vae16_fb0p0_film").graph_ae.eval()
    nf = torch.randn(2, 256, 128)
    mask = torch.ones(2, 256, dtype=torch.bool); mask[:, 200:] = False
    nf_garbage = nf.clone(); nf_garbage[:, 200:] = 50.0
    with torch.no_grad():
        _, zg, *_ = gae.encode(DenseGraphBatch(node_features=nf, edge_features=torch.empty(0), mask=mask), sample=False)
        _, zg_g, *_ = gae.encode(DenseGraphBatch(node_features=nf_garbage, edge_features=torch.empty(0), mask=mask), sample=False)
    assert torch.allclose(zg, zg_g, atol=1e-4)   # masked values do not leak into z_global


def test_encoder_zglobal_finite_when_dropped_values_are_nan():
    # NaN-safety regression: the neighborhood-AND-padding mask can leave a dropped node
    # with zero valid keys (self and every neighbor also dropped). Some GPU SDPA backends
    # return NaN for that all-masked softmax row, which then lands in node_features at the
    # dropped position. Simulate the artifact directly by seeding NaN at dropped positions
    # and require that it never reaches z_global (via the mask-aware stats pool AND the
    # nan_to_num safety net on the transformer output in GraphEncoder.forward).
    import torch
    from src.downstream.encode import _build_pl_module
    from src.data.components.graphs_datamodules import DenseGraphBatch
    gae = _build_pl_module("vae16_fb0p0_film").graph_ae.eval()
    nf = torch.randn(2, 256, 128)
    mask = torch.ones(2, 256, dtype=torch.bool); mask[:, 200:] = False  # >=1 valid node remains
    nf_nan = nf.clone(); nf_nan[:, 200:] = float("nan")
    with torch.no_grad():
        _, zg, *_ = gae.encode(DenseGraphBatch(node_features=nf_nan, edge_features=torch.empty(0), mask=mask), sample=False)
    assert torch.isfinite(zg).all()


def test_encoder_attention_all_true_mask_matches_mask_none_pin():
    # Direct pin at the transformer level (isolated from GraphEncoder/stats-pool wiring):
    # the new `neighborhood AND padding` is_encoder branch, given an all-True CLS-padded
    # mask, must reproduce the old `mask=None` (pure neighborhood mask) path exactly.
    import torch
    from src.downstream.encode import _build_pl_module
    gae = _build_pl_module("vae16_fb0p0_film").graph_ae.eval()
    transformer = gae.encoder.graph_transformer
    hidden_dim = gae.encoder.summary_node.shape[-1]
    n = 257  # 256 content nodes (16x16 grid) + 1 CLS
    x = torch.randn(2, n, hidden_dim)
    all_true = torch.ones(2, n, dtype=torch.bool)
    with torch.no_grad():
        out_masked = transformer(x, mask=all_true, is_encoder=True)
        out_none = transformer(x, mask=None, is_encoder=True)
    assert torch.allclose(out_masked, out_none, atol=1e-6)


def test_contrastive_path_finite_loss_and_encoder_gradient():
    from src.downstream.encode import _build_pl_module
    from src.models.components.contrastive import drop_views, ProjectionHead
    from src.models.components.losses import ContrastiveLoss
    pl = _build_pl_module("vae16_fb0p0_film")
    gae = pl.graph_ae.train()
    # Batch size must be a multiple of permuter.num_permutations (= data.hparams.num_aug_per_sample,
    # stubbed to 8 by `_build_pl_module`): NodeBottleneckEncoder.forward(sample=True) reshapes
    # eps via `node_features.shape[0] // self.num_permutations` when tiling VAE reparameterization
    # noise across augmented views, which crashes (0-sized tensor) for a non-multiple batch size
    # such as the brief's original B=4. Bumped to 8 (the brief specifies B=4 verbatim).
    batch = DenseGraphBatch(
        node_features=torch.randn(8, 256, 128),
        edge_features=torch.empty(0),
        mask=torch.ones(8, 256, dtype=torch.bool),
    )
    head = ProjectionHead(in_dim=_zg_dim(gae, batch), hidden_dim=512, out_dim=128)
    ntx = ContrastiveLoss(temperature=0.2, num_aug_per_sample=2)
    feats = []
    for v in drop_views(batch, p=0.2, n=2, generator=torch.Generator().manual_seed(0)):
        _, zg, *_ = gae.encode(v, sample=True)
        feats.append(head(zg))
    loss = ntx(torch.cat(feats, dim=0))                     # [2B, 128]
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    enc_grad = sum(
        p.grad.abs().sum().item()
        for n, p in gae.named_parameters()
        if "encoder" in n and p.grad is not None
    )
    assert enc_grad > 0.0
