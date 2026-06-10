"""
Diagnostic inference script for PIGVAE (imc-pigvae-film branch).

Loads a checkpoint, runs a few validation batches, and prints a detailed report
covering every subsystem that could be dead or collapsed:
  - Latent space (mu, std, active dims, KLD per dim)
  - Node feature diversity across D4 augmentations (position-level, not mean-pooled)
  - Raw permuter scores before Sinkhorn (are they identical for all 8 views?)
  - Permuter predictions and confidence per augmentation slot
  - Activation stats in encoder transformer, permuter transformer, decoder
  - Equivariance proof (encoder(flip(x)) == flip(encoder(x)))
  - Within-augmentation inter-node similarity (are all positions collapsing?)
  - Decoder std diagnostic (is decoder std ~ 1/sqrt(N)?)

Usage:
    python scripts/diagnose_model.py \\
        --ckpt /path/to/checkpoint.ckpt \\
        [--num-batches 3] \\
        [--batch-idx 5]        # pick a specific batch instead
        [--split val]          # val (default) or test
        [--tau 0.5]            # temperature (default: from checkpoint)
        [--data-dir /path/to/data]
        [--paths szary]        # paths config name
"""

import argparse
import operator
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── project root setup ──────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import rootutils
rootutils.setup_root(str(project_root), indicator=".project-root", pythonpath=True)

from omegaconf import OmegaConf, DictConfig
OmegaConf.register_new_resolver("multiply", lambda x, y: operator.mul(int(x), int(y)), replace=True)
OmegaConf.register_new_resolver("divide", lambda x, y: int(x) // int(y), replace=True)

from hydra.utils import instantiate

# ── helpers ─────────────────────────────────────────────────────────────────

NUM_VIEWS = 8
AUG_NAMES = ["r0_f", "r0_nf", "r180_f", "r180_nf", "r270_f", "r270_nf", "r90_f", "r90_nf"]


def _stat(t: torch.Tensor) -> dict:
    t = t.detach().float()
    return {
        "mean": round(t.mean().item(), 5),
        "std": round(t.std().item(), 5),
        "min": round(t.min().item(), 5),
        "max": round(t.max().item(), 5),
        "abs_max": round(t.abs().max().item(), 5),
    }


def make_activation_hook(store: dict, key: str, keep_tensor: bool = False):
    def hook(module, inp, output):
        if isinstance(output, torch.Tensor):
            t = output.detach().float()
            store[key] = _stat(t)
            if keep_tensor:
                store[key + "__tensor"] = t.cpu()
    return hook


# ── core diagnostics ─────────────────────────────────────────────────────────

def run_diagnostics(model, batch, tau: float) -> dict:
    model.eval()
    act_store = {}
    hooks = []

    ae = model.graph_ae

    # Register activation hooks (transformer decoder replaces CNN decoder)
    hook_targets = [
        ("encoder/graph_transformer", ae.encoder.graph_transformer),
        ("encoder/fc_in",             ae.encoder.fc_in),
        ("decoder/fc_in",             ae.decoder.fc_in),
        ("decoder/graph_transformer", ae.decoder.graph_transformer),
    ]
    if hasattr(ae.decoder, "node_fc_out"):
        hook_targets.append(("decoder/node_fc_out", ae.decoder.node_fc_out))
    if hasattr(ae, "permuter") and hasattr(ae.permuter, "scoring_fc"):
        hook_targets.append(("permuter/scoring_fc", ae.permuter.scoring_fc))
        hook_targets.append(("permuter/graph_transformer", ae.permuter.graph_transformer))

    for name, module in hook_targets:
        keep = (name == "permuter/scoring_fc")
        hooks.append(module.register_forward_hook(make_activation_hook(act_store, name, keep_tensor=keep)))

    with torch.no_grad():
        graph_emb, graph_pred, soft_probs, perm, mu, logvar = model(
            graph=batch, training=False, tau=tau
        )
        # node_features are not exposed through PLGraphAE.forward — call encode directly.
        # encode returns (z_nodes, z_global, node_features, mu, logvar).
        _, _, node_features, _, _ = model.graph_ae.encode(batch)

    for h in hooks:
        h.remove()

    total_B = node_features.shape[0]
    B = total_B // NUM_VIEWS
    N = node_features.shape[1]
    D = node_features.shape[2]

    # ── 1. Latent space ────────────────────────────────────────────────────
    if mu is not None and logvar is not None:
        std = (0.5 * logvar).exp()
        kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - std.pow(2))
        kld_per_dim_mean = kld_per_dim.mean(dim=0)
        active = (kld_per_dim_mean > 0.1).sum().item()
        latent_diag = {
            "vae":                "enabled",
            "mu_mean":            round(mu.mean().item(), 5),
            "mu_std":             round(mu.std().item(), 5),
            "std_mean":           round(std.mean().item(), 5),
            "std_per_dim_min":    round(std.mean(0).min().item(), 5),
            "std_per_dim_max":    round(std.mean(0).max().item(), 5),
            "active_dims":        active,
            "total_dims":         mu.shape[-1],
            "kld_per_dim_median": round(kld_per_dim_mean.median().item(), 5),
            "kld_per_dim_p95":    round(kld_per_dim_mean.quantile(0.95).item(), 5),
            "kld_per_dim_max":    round(kld_per_dim_mean.max().item(), 5),
            "total_kld_mean":     round(kld_per_dim.mean().item(), 5),
        }
    else:
        # Deterministic AE (vae=false) — report graph_emb stats instead
        z = graph_emb.detach().float()
        latent_diag = {
            "vae":            "disabled",
            "z_mean":         round(z.mean().item(), 5),
            "z_std":          round(z.std().item(), 5),
            "z_min":          round(z.min().item(), 5),
            "z_max":          round(z.max().item(), 5),
            "z_abs_max":      round(z.abs().max().item(), 5),
            "z_dims":         z.shape[-1],
            "z_norm_mean":    round(z.norm(dim=-1).mean().item(), 5),
        }

    # ── 2. Node feature diversity ─────────────────────────────────────────
    views = node_features.view(NUM_VIEWS, B, N, D)
    views_norm = F.normalize(views, dim=-1)

    sim_pos = torch.einsum("vbnd,ubnd->uvb", views_norm, views_norm) / N
    off_diag = ~torch.eye(NUM_VIEWS, dtype=torch.bool)

    views_mean = views.mean(dim=2)
    views_mean_norm = F.normalize(views_mean, dim=-1)
    sim_mean = torch.einsum("vbd,ubd->uvb", views_mean_norm, views_mean_norm)

    sim_pair = sim_pos.mean(dim=-1)

    node_diag = {
        "position_level_cos_sim (want low)": round(sim_pos[off_diag].mean().item(), 5),
        "position_level_cos_sim_std":        round(sim_pos[off_diag].std().item(), 5),
        "mean_pooled_cos_sim (always ~1)":   round(sim_mean[off_diag].mean().item(), 5),
        "feat_std_across_augs":              round(views.std(dim=0).mean().item(), 5),
        "feat_mean":                         round(node_features.mean().item(), 5),
        "feat_std":                          round(node_features.std().item(), 5),
        "feat_abs_max":                      round(node_features.abs().max().item(), 5),
        "pairwise_similarity": {
            f"{AUG_NAMES[i]}↔{AUG_NAMES[j]}": round(sim_pair[i, j].item(), 4)
            for i in range(NUM_VIEWS) for j in range(i + 1, NUM_VIEWS)
        },
    }

    # ── 3. Permuter diagnostics ──────────────────────────────────────────
    perm_diag = {"permuter": "oracle / disabled — no active permuter on this branch"}
    if soft_probs is not None and soft_probs.numel() > 0:
        probs = soft_probs.view(NUM_VIEWS, B, NUM_VIEWS)
        pred_classes = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        per_row_H = -(probs * (probs + 1e-8).log()).sum(-1)

        raw_scores = act_store.get("permuter/scoring_fc__tensor")
        score_diag = {}
        if raw_scores is not None:
            scores_by_view = raw_scores.view(NUM_VIEWS, B, NUM_VIEWS)
            max_score_diff = (
                scores_by_view.unsqueeze(0) - scores_by_view.unsqueeze(1)
            ).abs().max().item()
            mean_score_diff = (
                scores_by_view.unsqueeze(0) - scores_by_view.unsqueeze(1)
            ).abs().mean().item()
            img0_scores = scores_by_view[:, 0, :]
            img0_diffs  = img0_scores - img0_scores[0:1, :]
            score_diag = {
                "raw_scores_mean":       round(raw_scores.mean().item(), 5),
                "raw_scores_std":        round(raw_scores.std().item(), 5),
                "max_diff_across_views": round(max_score_diff, 6),
                "mean_diff_across_views": round(mean_score_diff, 6),
                "IMAGE_0_score_vectors": {
                    AUG_NAMES[v]: [round(x, 5) for x in img0_scores[v].tolist()]
                    for v in range(NUM_VIEWS)
                },
                "IMAGE_0_diff_vs_r0_f": {
                    AUG_NAMES[v]: [round(x, 7) for x in img0_diffs[v].tolist()]
                    for v in range(NUM_VIEWS)
                },
            }

        perm_diag = {
            "mean_confidence":       round(confidence.mean().item(), 5),
            "uniform_confidence_is": round(1.0 / NUM_VIEWS, 5),
            "per_row_entropy_mean":  round(per_row_H.mean().item(), 5),
            "max_possible_entropy":  round(np.log(NUM_VIEWS), 5),
            "all_views_same_class":  bool((pred_classes == pred_classes[0:1]).all().item()),
            "is_bijective":          bool(len({pred_classes[:, 0].tolist()[v] for v in range(NUM_VIEWS)}) == NUM_VIEWS),
            "predicted_class_per_aug": {
                AUG_NAMES[v]: int(pred_classes[v].mode().values.item())
                for v in range(NUM_VIEWS)
            },
            "confidence_per_aug": {
                AUG_NAMES[v]: round(confidence[v].mean().item(), 5)
                for v in range(NUM_VIEWS)
            },
            "score_diagnostics": score_diag,
        }

    # ── 4. Activation stats ───────────────────────────────────────────────
    activation_diag = {k: v for k, v in act_store.items() if not k.endswith("__tensor")}

    # ── 5. Decoder std diagnostic ─────────────────────────────────────────
    # graph_pred is a DenseGraphBatch; pull node_features tensor
    if hasattr(graph_pred, "node_features"):
        pred = graph_pred.node_features
    elif isinstance(graph_pred, (list, tuple)):
        pred = graph_pred[0]
    else:
        pred = graph_pred
    if pred.dim() == 3:
        decoder_std_diag = {
            "pred_shape": list(pred.shape),
            "global_std (1/sqrt(N) ~ 0.167 if collapsed)": round(pred.float().std().item(), 5),
            "per_position_std_mean": round(pred.float().std(dim=-1).mean().item(), 5),
            "per_position_std_min":  round(pred.float().std(dim=-1).min().item(), 5),
            "per_position_std_max":  round(pred.float().std(dim=-1).max().item(), 5),
            "across_position_std (should be > 0 if decoder is position-aware)":
                round(pred.float().mean(dim=-1).std(dim=-1).mean().item(), 5),
            "expected_collapsed_std": round(1.0 / N**0.5, 5),
        }
    else:
        decoder_std_diag = {"pred_shape": list(pred.shape), "note": "unexpected shape"}

    # ── 6. Equivariance proof ─────────────────────────────────────────────
    grid = 6
    flip_perm = [row * grid + (grid - 1 - col)
                 for row in range(grid) for col in range(grid)]

    h_identity = node_features[B].cpu()
    h_flip     = node_features[0].cpu()
    h_identity_permuted = h_identity[flip_perm, :]

    raw_diff  = (h_flip - h_identity).abs()
    perm_diff = (h_flip - h_identity_permuted).abs()

    cos_before = F.cosine_similarity(h_flip, h_identity, dim=-1).mean().item()
    cos_after  = F.cosine_similarity(h_flip, h_identity_permuted, dim=-1).mean().item()

    equivariance_diag = {
        "claim": "encoder(flip(x)) == flip(encoder(x)) iff encoder is permutation-equivariant",
        "diff WITHOUT permutation": {
            "max":    round(raw_diff.max().item(), 6),
            "mean":   round(raw_diff.mean().item(), 6),
            "cos_sim": round(cos_before, 6),
        },
        "diff AFTER applying flip_perm to identity": {
            "max":    round(perm_diff.max().item(), 6),
            "mean":   round(perm_diff.mean().item(), 6),
            "cos_sim": round(cos_after, 6),
        },
        "verdict": (
            "EQUIVARIANT (diff<1e-3 after perm)"
            if perm_diff.max().item() < 1e-3
            else "NOT perfectly equivariant or content collapsed"
        ),
    }

    # ── 7. Within-augmentation inter-node similarity ──────────────────────
    h_id_all  = node_features[B].cpu()
    h_id_norm = F.normalize(h_id_all, dim=-1)
    sim_within = torch.mm(h_id_norm, h_id_norm.t())
    off_diag_N = ~torch.eye(N, dtype=torch.bool)
    inter_node_cos = sim_within[off_diag_N]

    within_aug_diag = {
        "question": "are h_i and h_j similar for different positions i≠j in same augmentation?",
        "inter_node_cos_sim_mean": round(inter_node_cos.mean().item(), 6),
        "inter_node_cos_sim_min":  round(inter_node_cos.min().item(), 6),
        "inter_node_cos_sim_max":  round(inter_node_cos.max().item(), 6),
        "interpretation": (
            "COLLAPSED (encoder ignores which cell is at each position)"
            if inter_node_cos.mean().item() > 0.99
            else "OK (encoder produces position-specific representations)"
        ),
        "spot_check cos(h_0, h_17)": round(sim_within[0, 17].item(), 6),
        "spot_check cos(h_0, h_35)": round(sim_within[0, 35].item(), 6),
        "spot_check cos(h_5, h_30)": round(sim_within[5, 30].item(), 6),
    }

    # ── 8. Node inspection ───────────────────────────────────────────────
    node_inspection = {}
    for node_idx in [0, 5, 17, 35]:
        v_identity = h_identity[node_idx]
        v_flip     = h_flip[node_idx]
        cos = F.cosine_similarity(v_identity.unsqueeze(0), v_flip.unsqueeze(0)).item()
        l2  = (v_identity - v_flip).norm().item()
        node_inspection[f"node_{node_idx}"] = {
            "r0_nf (identity) first 16 dims": [round(x, 4) for x in v_identity[:16].tolist()],
            "r0_f  (flip)     first 16 dims": [round(x, 4) for x in v_flip[:16].tolist()],
            "element-wise diff first 16":     [round(x, 4) for x in (v_flip - v_identity)[:16].tolist()],
            "cosine_similarity":              round(cos, 6),
            "L2_norm_of_diff":               round(l2, 6),
            "L2_norm_of_vector":             round(v_identity.norm().item(), 4),
        }

    return {
        "latent":                            latent_diag,
        "node_features":                     node_diag,
        "permuter":                          perm_diag,
        "activations":                       activation_diag,
        "decoder_std_diagnostic":            decoder_std_diag,
        "equivariance_proof":                equivariance_diag,
        "within_aug_inter_node_similarity":  within_aug_diag,
        "node_inspection (image 0, r0_nf vs r0_f)": node_inspection,
    }


# ── pretty printer ────────────────────────────────────────────────────────────

def _print_dict(d: dict, indent: int = 0):
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            _print_dict(v, indent + 1)
        else:
            print(f"{pad}{k}: {v}")


def _print_section(title: str, data: dict, indent: int = 0):
    pad = "  " * indent
    print(f"\n{pad}{'─' * (60 - 2*indent)}")
    print(f"{pad}  {title}")
    print(f"{pad}{'─' * (60 - 2*indent)}")
    _print_dict(data, indent + 1)


def print_report(results: dict, batch_idx: int):
    print(f"\n{'═' * 62}")
    print(f"  PIGVAE DIAGNOSTIC REPORT — batch {batch_idx}")
    print(f"{'═' * 62}")

    _print_section("LATENT SPACE", results["latent"])
    _print_section("NODE FEATURE DIVERSITY (across D4 augmentations)", results["node_features"])
    _print_section("PERMUTER", results["permuter"])
    _print_section("ACTIVATION STATS (per module)", results["activations"])
    _print_section("DECODER STD DIAGNOSTIC (std ~ 1/sqrt(N) = collapsed)", results["decoder_std_diagnostic"])
    _print_section("EQUIVARIANCE PROOF (image 0: r0_nf vs r0_f)", results["equivariance_proof"])
    _print_section("WITHIN-AUG INTER-NODE SIMILARITY (cos(h_i, h_j) for i≠j?)",
                   results["within_aug_inter_node_similarity"])
    _print_section("NODE INSPECTION — actual vectors at 4 positions (image 0: r0_nf vs r0_f)",
                   results["node_inspection (image 0, r0_nf vs r0_f)"])


# ── inspection orchestrator ──────────────────────────────────────────────────

from src.models.components.llama_graph_transformer import (
    SelfAttention,
    TransformerBlock,
    get_full_mask,
    get_neighborhood_mask,
)
from src.utils.inspection import (
    attention_entropy_from_input,
    collect_activation_stats,
    film_diagnostics,
    image_space_reconstruction,
    latent_diagnostics,
    reconstruction_diagnostics,
    weight_diagnostics,
    write_report,
)


def inspect_model(model, batch, out_dir, meta: dict, pca_layer=None) -> dict:
    """Orchestrate all inspection sections on a GraphAE + one batch.

    model: GraphAE (pl_module.graph_ae)
    batch: DenseGraphBatch already on the correct device
    out_dir: path-like; artifacts are written here
    meta: dict of run metadata (ckpt path, split, tau, …)
    """
    model.eval()
    results: dict = {"meta": meta}

    # A. Weights (no data needed)
    results["weights"] = weight_diagnostics(model)

    # B. Activations: hook transformer blocks + graph_transformers + key linear layers.
    #    Run the FULL encode+decode path so decoder blocks are also captured.
    def act_filter(name: str, m: nn.Module) -> bool:
        return (
            isinstance(m, TransformerBlock)
            or name.endswith("graph_transformer")
            or (isinstance(m, nn.Linear) and ("fc_out" in name or "projection_in" in name))
        )

    def _full_forward() -> None:
        zn, zg, _nf, _mu, _lv = model.encode(batch)
        model.decode(zn, zg, batch.mask)

    results["activations"] = collect_activation_stats(model, _full_forward, act_filter)

    # C. Attention entropy — capture each SelfAttention input via pre-hook,
    #    then recompute weights with the correct mask (encoder=neighborhood, decoder=full).
    captured: dict = {}
    pre_handles = []
    for _name, _mod in model.named_modules():
        if isinstance(_mod, SelfAttention):
            def _pre_hook(_m, args, _n=_name):
                captured[_n] = args[0].detach()
            pre_handles.append(_mod.register_forward_pre_hook(_pre_hook))
    with torch.no_grad():
        z_nodes, z_global, _node_features, _mu, _logvar = model.encode(batch)
        graph_pred = model.decode(z_nodes, z_global, batch.mask)
    for h in pre_handles:
        h.remove()

    attn: dict = {}
    for name, mod in model.named_modules():
        if isinstance(mod, SelfAttention) and name in captured:
            x = captured[name]
            n_seq = x.shape[1]
            is_enc = name.startswith("encoder")
            if is_enc:
                mask = get_neighborhood_mask(n_seq, is_enc, x.device)
            else:
                mask = get_full_mask(batch.mask, False, x.device)
            attn[name] = attention_entropy_from_input(mod, x, mask)
    results["attention"] = attn

    # D. Latent bottleneck health
    results["latent"] = latent_diagnostics(z_nodes)

    # D2. FiLM conditioning magnitudes (confirms tanh-bounding fired; |gamma|<=1 if bound)
    film_stats = film_diagnostics(model.decoder, z_global)
    if film_stats:
        results["film"] = film_stats

    # E. Reconstruction quality (PCA-coefficient space)
    num_views = getattr(model.permuter, "num_permutations", 8)
    results["reconstruction"] = reconstruction_diagnostics(
        graph_pred.node_features, batch.node_features, num_views
    )
    # E2. Image-space (inverse-PCA) reconstruction — the metric the real pipeline cares about.
    # input_features (raw pre-PCA x), if kept by the collator, also gives error vs the input.
    if pca_layer is not None:
        results["reconstruction"]["image_space"] = image_space_reconstruction(
            graph_pred.node_features, batch.node_features, pca_layer,
            input_x=getattr(batch, "input_features", None),
        )

    write_report(results, out_dir)
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def load_model_and_data(
    ckpt_path: str,
    data_dir: str | None,
    split: str,
    paths_name: str = "szary",
    experiment: str | None = None,
):
    configs = project_root / "configs"

    model_cfg = OmegaConf.load(configs / "model" / "model.yaml")
    data_cfg  = OmegaConf.load(configs / "data"  / "mnist.yaml")
    paths_cfg = OmegaConf.load(configs / "paths" / f"{paths_name}.yaml")

    for name, fn in [("multiply", lambda x, y: int(x) * int(y)),
                     ("divide",   lambda x, y: int(x) // int(y))]:
        try:
            OmegaConf.register_new_resolver(name, fn)
        except Exception:
            pass

    if data_dir:
        OmegaConf.update(paths_cfg, "data_dir", data_dir, merge=True)

    trainer_stub = OmegaConf.create({"max_epochs": 200, "min_epochs": 1})
    data_stub    = OmegaConf.create({"hparams": {"num_aug_per_sample": 8, "batch_size": 16}})

    # Overlay an experiment's `model` (and `trainer`/`data`) overrides so architecture-varying
    # checkpoints (different input_size / neighborhood_radius / pos_bias / num_node_features / ...)
    # build the matching model and load cleanly, and the dataloader emits matching features.
    exp_data = None
    if experiment:
        exp_cfg = OmegaConf.load(configs / "experiment" / f"{experiment}.yaml")
        if "model" in exp_cfg:
            model_cfg = OmegaConf.merge(model_cfg, exp_cfg.model)
        if "trainer" in exp_cfg:
            trainer_stub = OmegaConf.merge(trainer_stub, exp_cfg.trainer)
        if "data" in exp_cfg:
            exp_data = exp_cfg.data  # e.g. num_pca_components/num_node_features=192
        print(f"[info] applied overrides from experiment={experiment}")

    ctx = OmegaConf.create({
        "model":   model_cfg,
        "data":    data_stub,
        "trainer": trainer_stub,
        "paths":   paths_cfg,
    })
    OmegaConf.set_struct(ctx, False)
    ctx.model = model_cfg

    data_cfg_resolved = OmegaConf.merge(data_cfg, exp_data) if exp_data is not None else OmegaConf.merge(data_cfg, {})
    OmegaConf.update(data_cfg_resolved, "hparams.data_dir", OmegaConf.select(ctx, "paths.data_dir"))

    dm = instantiate(data_cfg_resolved)
    dm.setup(stage="fit" if split == "val" else "test")
    dataloader = dm.val_dataloader() if split == "val" else dm.test_dataloader()
    # Keep the raw pre-PCA input on inspection batches so we can measure error vs the input x.
    # Set before iterating so it propagates to dataloader workers. Training is unaffected.
    collate_fn = getattr(dataloader, "collate_fn", None)
    if collate_fn is not None and hasattr(collate_fn, "keep_input"):
        collate_fn.keep_input = True

    graph_ae      = instantiate(ctx.model.graph_ae)
    critic        = instantiate(ctx.model.critic)
    temp_sched    = instantiate(ctx.model.temperature_scheduler)
    entropy_sched = instantiate(ctx.model.entropy_weight_scheduler)
    kld_sched     = instantiate(ctx.model.kld_alpha_scheduler)
    optimizer_fn  = instantiate(ctx.model.optimizer)
    scheduler_cfg = ctx.model.scheduler

    from src.models.pigvae_auto_module import PLGraphAE
    model = PLGraphAE(
        graph_ae=graph_ae,
        critic=critic,
        temperature_scheduler=temp_sched,
        entropy_weight_scheduler=entropy_sched,
        kld_alpha_scheduler=kld_sched,
        optimizer=optimizer_fn,
        scheduler=scheduler_cfg,
        compile=False,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    epoch = ckpt.get("epoch", 0)
    tau = temp_sched(epoch)
    print(f"[info] Loaded checkpoint — epoch {epoch}, tau={tau:.4f}")

    model.eval()
    # pca_layer (built by the collator during val_dataloader) enables image-space inverse-PCA metrics
    return model, dataloader, tau, getattr(dm, "pca_layer", None)


def main():
    parser = argparse.ArgumentParser(description="PIGVAE diagnostic inference (film branch)")
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    parser.add_argument("--data-dir", default=None, help="Override data_dir")
    parser.add_argument("--paths", default="szary", help="Paths config name")
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment config name (configs/experiment/<name>.yaml) whose `model` overrides "
             "are applied so an architecture-varying checkpoint builds the matching model.",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--batch-idx", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Run full weight/activation/attention/latent/reconstruction inspection",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output dir for inspection artifacts (default: logs/diagnostics/<ckpt-stem>)",
    )
    args = parser.parse_args()

    model, dataloader, tau, pca_layer = load_model_and_data(
        args.ckpt, args.data_dir, args.split, args.paths, args.experiment
    )
    if args.tau is not None:
        tau = args.tau
        print(f"[info] tau overridden to {tau}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if args.inspect:
        out_dir = args.out_dir or f"logs/diagnostics/{Path(args.ckpt).stem}"
        meta = {
            "ckpt": args.ckpt,
            "split": args.split,
            "tau": tau,
            "run": Path(args.ckpt).parent.name,
        }
        inspect_batch = next(iter(dataloader)).to(device)
        graph_ae = model.graph_ae if hasattr(model, "graph_ae") else model
        inspect_model(graph_ae, inspect_batch, out_dir, meta, pca_layer)
        print(f"[inspect] wrote artifacts to {out_dir}")

    n_run = 0
    for batch_i, batch in enumerate(dataloader):
        if args.batch_idx is not None and batch_i != args.batch_idx:
            continue

        batch = batch.to(device)
        results = run_diagnostics(model, batch, tau)
        print_report(results, batch_idx=batch_i)

        n_run += 1
        if args.batch_idx is not None:
            break
        if n_run >= args.num_batches:
            break

    if n_run == 0:
        print(f"[ERROR] --batch-idx {args.batch_idx} was not reached.")
        sys.exit(1)


if __name__ == "__main__":
    main()
