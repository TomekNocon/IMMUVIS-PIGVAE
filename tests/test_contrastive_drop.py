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


def test_combine_neighborhood_and_padding_mask_keeps_self_key_for_isolated_node():
    # Direct pin on the REAL production helper (`combine_neighborhood_and_padding_mask`,
    # extracted from SelfAttention.forward's `elif is_encoder:` branch so it's testable
    # in isolation, not reimplemented here). Construct the worst case: a query `q` whose
    # entire radius-1 neighborhood (including itself) is invalid in `pad`. Before the
    # fix this query's row was `neigh & pad` alone -> all-False (every key masked out,
    # a softmax-NaN hazard on some SDPA backends -- not reproducible with a real forward
    # pass on this CPU/MATH backend, which zero-fills instead; see the encode()-level
    # test below for the end-to-end path). The fix ORs in an identity matrix so every
    # row keeps >=1 True key.
    from src.models.components.llama_graph_transformer import (
        combine_neighborhood_and_padding_mask,
        get_neighborhood_mask,
    )
    num_nodes = 257  # 16x16 grid + CLS at index 0
    neigh = get_neighborhood_mask(num_nodes, is_encoder=True, device="cpu", radius=1)
    q = 137  # mask index of content node (row=8, col=8): 8*16 + 8 + 1 (CLS offset)
    neighbor_idx = neigh[q].nonzero(as_tuple=True)[0]  # q's radius-1 neighborhood (incl. self)
    assert neighbor_idx.numel() == 5  # interior node: self + 4 grid neighbors

    pad = torch.ones(1, num_nodes, dtype=torch.bool)
    pad[0, neighbor_idx] = False  # drop q AND its entire neighborhood

    attn_mask = combine_neighborhood_and_padding_mask(neigh, pad)  # [1, 1, N, N]
    attn_mask = attn_mask[:, 0]                                    # [1, N, N]
    assert attn_mask[0].any(dim=-1).all()  # every query row keeps >=1 True key
    assert attn_mask[0, q, q]              # specifically, q's own self-key is restored
    # no leakage: pre-fix (`neigh & pad`) is untouched off-diagonal; only a row's own
    # diagonal entry can flip False->True, so q is still excluded as a key for every
    # OTHER query -- e.g. CLS (row 0) still cannot attend to the dropped node q.
    pre_fix = neigh.unsqueeze(0) & pad.unsqueeze(1)
    off_diag = ~torch.eye(num_nodes, dtype=torch.bool)
    assert torch.equal(attn_mask[0][off_diag], pre_fix[0][off_diag])
    assert not attn_mask[0, 0, q]  # CLS still cannot attend to dropped node q as a key


def test_encoder_zglobal_finite_when_node_and_full_neighborhood_dropped():
    # Integration-level regression: on the REAL encoder (real `_build_pl_module`
    # graph_ae, real mask-aware attention wiring), drop one content node AND its
    # entire radius-1 neighborhood via the `mask` argument to `encode`. Before the
    # self-key fix this leaves an all-False attention row for that node -> NaN
    # softmax row -> `0 * NaN == NaN` leaks into the CLS row of the stats pool ->
    # z_global loses its CLS contribution (silently, no crash). After the fix the
    # isolated node still attends to itself, so no NaN is ever produced.
    import torch
    from src.downstream.encode import _build_pl_module
    from src.data.components.graphs_datamodules import DenseGraphBatch
    from src.models.components.llama_graph_transformer import get_neighborhood_mask

    gae = _build_pl_module("vae16_fb0p0_film").graph_ae.eval()
    num_nodes = 257  # 16x16 grid + CLS
    neigh = get_neighborhood_mask(num_nodes, is_encoder=True, device="cpu", radius=1)
    q = 137  # interior content node (row=8, col=8)
    neighbor_idx = neigh[q].nonzero(as_tuple=True)[0]

    nf = torch.randn(2, 256, 128)
    mask = torch.ones(2, num_nodes - 1, dtype=torch.bool)
    # neighbor_idx is in CLS-padded (257-wide) coordinates; content mask excludes CLS at 0.
    content_idx = neighbor_idx - 1
    mask[:, content_idx] = False
    with torch.no_grad():
        _, zg, *_ = gae.encode(
            DenseGraphBatch(node_features=nf, edge_features=torch.empty(0), mask=mask),
            sample=False,
        )
    assert torch.isfinite(zg).all()


def _build_pl_module_with_contrastive(experiment: str, **overrides):
    """Same Hydra composition as `_build_pl_module` (src/downstream/encode.py), but
    forwards contrastive kwargs straight to `PLGraphAE.__init__`.

    `_build_pl_module` itself never forwards `contrastive_loss_scale`/`drop_p`/etc.
    (it hardcodes the `PLGraphAE(...)` call without them), so there is no way to get
    a real, registered `projection_head`/`contrastive_loss` out of it. This mirrors
    its composition logic locally instead of modifying that helper (out of scope
    here), so tests can exercise `PLGraphAE._drop_contrastive` / the
    `contrastive_loss_scale` gate on an actual module instance rather than
    hand-built components.
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from src.downstream.encode import _PROJECT_ROOT
    from src.models.pigvae_auto_module import PLGraphAE

    configs = _PROJECT_ROOT / "configs"
    for name, fn in [("multiply", lambda x, y: int(x) * int(y)), ("divide", lambda x, y: int(x) // int(y))]:
        try:
            OmegaConf.register_new_resolver(name, fn)
        except Exception:
            pass
    model_cfg = OmegaConf.load(configs / "model" / "model.yaml")
    exp_cfg = OmegaConf.load(configs / "experiment" / f"{experiment}.yaml")
    if "model" in exp_cfg:
        model_cfg = OmegaConf.merge(model_cfg, exp_cfg.model)
    trainer_stub = OmegaConf.create({"max_epochs": 200, "min_epochs": 1})
    if "trainer" in exp_cfg:
        trainer_stub = OmegaConf.merge(trainer_stub, exp_cfg.trainer)
    data_stub = OmegaConf.create({"hparams": {"num_aug_per_sample": 8, "batch_size": 16}})
    if "data" in exp_cfg:
        data_stub = OmegaConf.merge(data_stub, exp_cfg.data)
    ctx = OmegaConf.create({"model": model_cfg, "trainer": trainer_stub, "data": data_stub})
    OmegaConf.set_struct(ctx, False)
    return PLGraphAE(
        graph_ae=instantiate(ctx.model.graph_ae),
        critic=instantiate(ctx.model.critic),
        temperature_scheduler=instantiate(ctx.model.temperature_scheduler),
        entropy_weight_scheduler=instantiate(ctx.model.entropy_weight_scheduler),
        kld_alpha_scheduler=instantiate(ctx.model.kld_alpha_scheduler),
        optimizer=instantiate(ctx.model.optimizer),
        scheduler=ctx.model.scheduler,
        compile=False,
        **overrides,
    )


def test_drop_contrastive_gate_on_real_pl_module():
    # Exercises the code actually added to training_step: contrastive_loss_scale>0
    # gate, drop_views -> graph_ae.encode -> self.projection_head -> self.contrastive_loss
    # on the REGISTERED submodules of a real PLGraphAE instance (not hand-built stand-ins).
    pl = _build_pl_module_with_contrastive(
        "vae16_fb0p0_film", contrastive_loss_scale=0.05, drop_p=0.2
    )
    assert hasattr(pl, "projection_head")
    assert hasattr(pl, "contrastive_loss")
    pl.graph_ae.train()
    batch = DenseGraphBatch(
        node_features=torch.randn(8, 256, 128),
        edge_features=torch.empty(0),
        mask=torch.ones(8, 256, dtype=torch.bool),
    )
    contrastive = pl._drop_contrastive(batch)
    assert contrastive is not None
    assert torch.isfinite(contrastive) and contrastive.item() > 0.0
    contrastive.backward()
    enc_grad = sum(
        p.grad.abs().sum().item()
        for n, p in pl.graph_ae.named_parameters()
        if "encoder" in n and p.grad is not None
    )
    assert enc_grad > 0.0


def test_drop_contrastive_gate_off_by_default():
    # Default contrastive_loss_scale=0.0 -> no projection_head/contrastive_loss
    # submodules registered, and _drop_contrastive is a no-op (returns None).
    pl = _build_pl_module_with_contrastive("vae16_fb0p0_film")
    assert not hasattr(pl, "projection_head")
    assert not hasattr(pl, "contrastive_loss")
    batch = DenseGraphBatch(
        node_features=torch.randn(8, 256, 128),
        edge_features=torch.empty(0),
        mask=torch.ones(8, 256, dtype=torch.bool),
    )
    assert pl._drop_contrastive(batch) is None


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
