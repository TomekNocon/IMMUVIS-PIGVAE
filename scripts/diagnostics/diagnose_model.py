"""Inspect a trained PIGVAE checkpoint for dead or collapsed subsystems.

Loads a checkpoint, runs a few validation batches and prints a report on:
  - Latent space (mu, std, active dims, KLD per dim)
  - Node feature diversity across D4 augmentations (position-level, not mean-pooled)
  - Permuter predictions and raw scores per augmentation slot (if the permuter is active)
  - Activation stats in encoder transformer, permuter transformer, decoder
  - Equivariance check (encoder(flip(x)) == flip(encoder(x)))
  - Within-augmentation inter-node similarity (are all positions collapsing?)
  - Decoder std diagnostic (is decoder std ~ 1/sqrt(N)?)
With ``inspect=true`` it also writes the full offline inspection report (weights, activations,
attention entropy, latent spectrum, FiLM, reconstruction) to ``out_dir``.

Usage:
    uv run python scripts/diagnostics/diagnose_model.py experiment=pigvae ckpt_path=/raid_.../last.ckpt
    uv run python scripts/diagnostics/diagnose_model.py experiment=pigvae ckpt_path=... inspect=true
    uv run python scripts/diagnostics/diagnose_model.py experiment=pigvae ckpt_path=... batch_idx=5 tau=0.5
"""

import sys
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig

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
from src.utils.resolvers import register_resolvers

register_resolvers()

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
    """Forward hook storing summary stats (and optionally the tensor) under ``key``."""

    def hook(module, inp, output):
        if isinstance(output, torch.Tensor):
            t = output.detach().float()
            store[key] = _stat(t)
            if keep_tensor:
                store[key + "__tensor"] = t.cpu()

    return hook


def _latent_section(graph_emb, mu, logvar) -> dict:
    """mu/std/KL statistics of the VAE latent, or z statistics of a deterministic AE."""
    if mu is None or logvar is None:
        z = graph_emb.detach().float()
        return {
            "vae": "disabled",
            "z_mean": round(z.mean().item(), 5),
            "z_std": round(z.std().item(), 5),
            "z_min": round(z.min().item(), 5),
            "z_max": round(z.max().item(), 5),
            "z_abs_max": round(z.abs().max().item(), 5),
            "z_dims": z.shape[-1],
            "z_norm_mean": round(z.norm(dim=-1).mean().item(), 5),
        }
    std = (0.5 * logvar).exp()
    kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - std.pow(2))
    kld_per_dim_mean = kld_per_dim.mean(dim=0)
    active = (kld_per_dim_mean > 0.1).sum().item()
    # Position on the AE<->VAE spectrum (law-of-total-variance overlap; the full breakdown is in
    # the `inspect=true` report, latent.spectrum).
    mu_flat = mu.reshape(-1, mu.shape[-1])
    std_flat = std.reshape(-1, std.shape[-1])
    mu_var = mu_flat.var(dim=0, unbiased=False).mean()
    sigma_sq = std_flat.pow(2).mean(dim=0).mean()
    overlap_ratio = (sigma_sq.sqrt() / mu_var.clamp_min(1e-12).sqrt()).item()
    noise_fraction = (sigma_sq / (mu_var + sigma_sq).clamp_min(1e-12)).item()
    return {
        "vae": "enabled",
        "mu_mean": round(mu.mean().item(), 5),
        "mu_std": round(mu.std().item(), 5),
        "std_mean": round(std.mean().item(), 5),
        "std_per_dim_min": round(std.mean(0).min().item(), 5),
        "std_per_dim_max": round(std.mean(0).max().item(), 5),
        "active_dims": active,
        "total_dims": mu.shape[-1],
        "kld_per_dim_median": round(kld_per_dim_mean.median().item(), 5),
        "kld_per_dim_p95": round(kld_per_dim_mean.quantile(0.95).item(), 5),
        "kld_per_dim_max": round(kld_per_dim_mean.max().item(), 5),
        "total_kld_mean": round(kld_per_dim.mean().item(), 5),
        "overlap_ratio": round(overlap_ratio, 4),
        "noise_fraction": round(noise_fraction, 4),
        "aggregate_var_mean": round((mu_var + sigma_sq).item(), 4),
    }


def _node_diversity_section(node_features: torch.Tensor) -> dict:
    """Cosine similarity of the encoder node features across the 8 D4 views."""
    total_b, N, D = node_features.shape
    B = total_b // NUM_VIEWS
    views = node_features.view(NUM_VIEWS, B, N, D)
    views_norm = F.normalize(views, dim=-1)
    sim_pos = torch.einsum("vbnd,ubnd->uvb", views_norm, views_norm) / N
    off_diag = ~torch.eye(NUM_VIEWS, dtype=torch.bool)
    views_mean_norm = F.normalize(views.mean(dim=2), dim=-1)
    sim_mean = torch.einsum("vbd,ubd->uvb", views_mean_norm, views_mean_norm)
    sim_pair = sim_pos.mean(dim=-1)
    return {
        "position_level_cos_sim (want low)": round(sim_pos[off_diag].mean().item(), 5),
        "position_level_cos_sim_std": round(sim_pos[off_diag].std().item(), 5),
        "mean_pooled_cos_sim (always ~1)": round(sim_mean[off_diag].mean().item(), 5),
        "feat_std_across_augs": round(views.std(dim=0).mean().item(), 5),
        "feat_mean": round(node_features.mean().item(), 5),
        "feat_std": round(node_features.std().item(), 5),
        "feat_abs_max": round(node_features.abs().max().item(), 5),
        "pairwise_similarity": {
            f"{AUG_NAMES[i]}↔{AUG_NAMES[j]}": round(sim_pair[i, j].item(), 4)
            for i in range(NUM_VIEWS)
            for j in range(i + 1, NUM_VIEWS)
        },
    }


def _permuter_section(soft_probs, raw_scores, B: int) -> dict:
    """Predicted D4 class, confidence and raw scores per view (only with an active permuter)."""
    if soft_probs is None or soft_probs.numel() == 0:
        return {"permuter": "oracle / disabled — no active permuter on this branch"}
    probs = soft_probs.view(NUM_VIEWS, B, NUM_VIEWS)
    pred_classes = probs.argmax(dim=-1)
    confidence = probs.max(dim=-1).values
    per_row_h = -(probs * (probs + 1e-8).log()).sum(-1)

    score_diag = {}
    if raw_scores is not None:
        scores_by_view = raw_scores.view(NUM_VIEWS, B, NUM_VIEWS)
        pairwise = (scores_by_view.unsqueeze(0) - scores_by_view.unsqueeze(1)).abs()
        img0_scores = scores_by_view[:, 0, :]
        img0_diffs = img0_scores - img0_scores[0:1, :]
        score_diag = {
            "raw_scores_mean": round(raw_scores.mean().item(), 5),
            "raw_scores_std": round(raw_scores.std().item(), 5),
            "max_diff_across_views": round(pairwise.max().item(), 6),
            "mean_diff_across_views": round(pairwise.mean().item(), 6),
            "IMAGE_0_score_vectors": {
                AUG_NAMES[v]: [round(x, 5) for x in img0_scores[v].tolist()]
                for v in range(NUM_VIEWS)
            },
            "IMAGE_0_diff_vs_r0_f": {
                AUG_NAMES[v]: [round(x, 7) for x in img0_diffs[v].tolist()]
                for v in range(NUM_VIEWS)
            },
        }
    return {
        "mean_confidence": round(confidence.mean().item(), 5),
        "uniform_confidence_is": round(1.0 / NUM_VIEWS, 5),
        "per_row_entropy_mean": round(per_row_h.mean().item(), 5),
        "max_possible_entropy": round(np.log(NUM_VIEWS), 5),
        "all_views_same_class": bool((pred_classes == pred_classes[0:1]).all().item()),
        "is_bijective": bool(len(set(pred_classes[:, 0].tolist())) == NUM_VIEWS),
        "predicted_class_per_aug": {
            AUG_NAMES[v]: int(pred_classes[v].mode().values.item()) for v in range(NUM_VIEWS)
        },
        "confidence_per_aug": {
            AUG_NAMES[v]: round(confidence[v].mean().item(), 5) for v in range(NUM_VIEWS)
        },
        "score_diagnostics": score_diag,
    }


def _decoder_std_section(graph_pred, N: int) -> dict:
    """Spread of the decoder output; a collapsed decoder has std ~ 1/sqrt(N)."""
    pred = graph_pred.node_features
    if pred.dim() != 3:
        return {"pred_shape": list(pred.shape), "note": "unexpected shape"}
    pred = pred.float()
    return {
        "pred_shape": list(pred.shape),
        "global_std (1/sqrt(N) ~ 0.167 if collapsed)": round(pred.std().item(), 5),
        "per_position_std_mean": round(pred.std(dim=-1).mean().item(), 5),
        "per_position_std_min": round(pred.std(dim=-1).min().item(), 5),
        "per_position_std_max": round(pred.std(dim=-1).max().item(), 5),
        "across_position_std (should be > 0 if decoder is position-aware)": round(
            pred.mean(dim=-1).std(dim=-1).mean().item(), 5
        ),
        "expected_collapsed_std": round(1.0 / N**0.5, 5),
    }


def _equivariance_section(h_identity: torch.Tensor, h_flip: torch.Tensor) -> dict:
    """Does flipping the input permute the encoder output the same way (image 0)?"""
    N = h_identity.shape[0]
    grid = round(N**0.5)
    if grid * grid != N:
        return {"note": f"skipped: N={N} is not a square grid"}
    flip_perm = [row * grid + (grid - 1 - col) for row in range(grid) for col in range(grid)]
    h_identity_permuted = h_identity[flip_perm, :]
    raw_diff = (h_flip - h_identity).abs()
    perm_diff = (h_flip - h_identity_permuted).abs()
    return {
        "claim": "encoder(flip(x)) == flip(encoder(x)) iff encoder is permutation-equivariant",
        "diff WITHOUT permutation": {
            "max": round(raw_diff.max().item(), 6),
            "mean": round(raw_diff.mean().item(), 6),
            "cos_sim": round(F.cosine_similarity(h_flip, h_identity, dim=-1).mean().item(), 6),
        },
        "diff AFTER applying flip_perm to identity": {
            "max": round(perm_diff.max().item(), 6),
            "mean": round(perm_diff.mean().item(), 6),
            "cos_sim": round(
                F.cosine_similarity(h_flip, h_identity_permuted, dim=-1).mean().item(), 6
            ),
        },
        "verdict": (
            "EQUIVARIANT (diff<1e-3 after perm)"
            if perm_diff.max().item() < 1e-3
            else "NOT perfectly equivariant or content collapsed"
        ),
    }


def _inter_node_section(h_identity: torch.Tensor) -> dict:
    """Cosine similarity between different positions of one view (collapse check)."""
    N = h_identity.shape[0]
    h_norm = F.normalize(h_identity, dim=-1)
    sim_within = torch.mm(h_norm, h_norm.t())
    inter_node_cos = sim_within[~torch.eye(N, dtype=torch.bool)]
    return {
        "question": "are h_i and h_j similar for different positions i≠j in same augmentation?",
        "inter_node_cos_sim_mean": round(inter_node_cos.mean().item(), 6),
        "inter_node_cos_sim_min": round(inter_node_cos.min().item(), 6),
        "inter_node_cos_sim_max": round(inter_node_cos.max().item(), 6),
        "interpretation": (
            "COLLAPSED (encoder ignores which cell is at each position)"
            if inter_node_cos.mean().item() > 0.99
            else "OK (encoder produces position-specific representations)"
        ),
        "spot_check cos(h_0, h_17)": round(sim_within[0, 17].item(), 6),
        "spot_check cos(h_0, h_35)": round(sim_within[0, 35].item(), 6),
        "spot_check cos(h_5, h_30)": round(sim_within[5, 30].item(), 6),
    }


def _node_inspection_section(h_identity: torch.Tensor, h_flip: torch.Tensor) -> dict:
    """The raw feature vectors of four positions, identity view vs flipped view."""
    out = {}
    for node_idx in [0, 5, 17, 35]:
        v_identity, v_flip = h_identity[node_idx], h_flip[node_idx]
        out[f"node_{node_idx}"] = {
            "r0_nf (identity) first 16 dims": [round(x, 4) for x in v_identity[:16].tolist()],
            "r0_f  (flip)     first 16 dims": [round(x, 4) for x in v_flip[:16].tolist()],
            "element-wise diff first 16": [
                round(x, 4) for x in (v_flip - v_identity)[:16].tolist()
            ],
            "cosine_similarity": round(
                F.cosine_similarity(v_identity.unsqueeze(0), v_flip.unsqueeze(0)).item(), 6
            ),
            "L2_norm_of_diff": round((v_identity - v_flip).norm().item(), 6),
            "L2_norm_of_vector": round(v_identity.norm().item(), 4),
        }
    return out


def run_diagnostics(model, batch, tau: float) -> dict:
    """Forward one batch with activation hooks and compute every report section."""
    model.eval()
    ae = model.graph_ae
    hook_targets = [
        ("encoder/graph_transformer", ae.encoder.graph_transformer),
        ("encoder/fc_in", ae.encoder.fc_in),
        ("decoder/fc_in", ae.decoder.fc_in),
        ("decoder/graph_transformer", ae.decoder.graph_transformer),
    ]
    if hasattr(ae.decoder, "node_fc_out"):
        hook_targets.append(("decoder/node_fc_out", ae.decoder.node_fc_out))
    if hasattr(ae, "permuter") and hasattr(ae.permuter, "scoring_fc"):
        hook_targets.append(("permuter/scoring_fc", ae.permuter.scoring_fc))
        hook_targets.append(("permuter/graph_transformer", ae.permuter.graph_transformer))

    act_store: dict = {}
    hooks = [
        module.register_forward_hook(
            make_activation_hook(act_store, name, keep_tensor=name == "permuter/scoring_fc")
        )
        for name, module in hook_targets
    ]
    with torch.no_grad():
        graph_emb, graph_pred, soft_probs, _perm, mu, logvar = model(
            graph=batch, training=False, tau=tau
        )
        # PLGraphAE.forward does not return the node features, so encode once more.
        _, _, node_features, _, _ = ae.encode(batch)
    for h in hooks:
        h.remove()

    B = node_features.shape[0] // NUM_VIEWS
    N = node_features.shape[1]
    # Image 0 in the identity view (r0_nf, block 1) and the flipped view (r0_f, block 0).
    h_identity = node_features[B].cpu()
    h_flip = node_features[0].cpu()
    return {
        "latent": _latent_section(graph_emb, mu, logvar),
        "node_features": _node_diversity_section(node_features),
        "permuter": _permuter_section(soft_probs, act_store.get("permuter/scoring_fc__tensor"), B),
        "activations": {k: v for k, v in act_store.items() if not k.endswith("__tensor")},
        "decoder_std_diagnostic": _decoder_std_section(graph_pred, N),
        "equivariance_proof": _equivariance_section(h_identity, h_flip),
        "within_aug_inter_node_similarity": _inter_node_section(h_identity),
        "node_inspection (image 0, r0_nf vs r0_f)": _node_inspection_section(h_identity, h_flip),
    }


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
    print(f"\n{pad}{'─' * (60 - 2 * indent)}")
    print(f"{pad}  {title}")
    print(f"{pad}{'─' * (60 - 2 * indent)}")
    _print_dict(data, indent + 1)


def print_report(results: dict, batch_idx: int):
    print(f"\n{'═' * 62}")
    print(f"  PIGVAE DIAGNOSTIC REPORT — batch {batch_idx}")
    print(f"{'═' * 62}")

    _print_section("LATENT SPACE", results["latent"])
    _print_section("NODE FEATURE DIVERSITY (across D4 augmentations)", results["node_features"])
    _print_section("PERMUTER", results["permuter"])
    _print_section("ACTIVATION STATS (per module)", results["activations"])
    _print_section(
        "DECODER STD DIAGNOSTIC (std ~ 1/sqrt(N) = collapsed)", results["decoder_std_diagnostic"]
    )
    _print_section("EQUIVARIANCE PROOF (image 0: r0_nf vs r0_f)", results["equivariance_proof"])
    _print_section(
        "WITHIN-AUG INTER-NODE SIMILARITY (cos(h_i, h_j) for i≠j?)",
        results["within_aug_inter_node_similarity"],
    )
    _print_section(
        "NODE INSPECTION — actual vectors at 4 positions (image 0: r0_nf vs r0_f)",
        results["node_inspection (image 0, r0_nf vs r0_f)"],
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

    # D. Latent bottleneck health (+ AE<->VAE spectrum when mu/logvar are available)
    results["latent"] = latent_diagnostics(z_nodes, mu=_mu, logvar=_logvar)

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
            graph_pred.node_features,
            batch.node_features,
            pca_layer,
            input_x=getattr(batch, "input_features", None),
        )

    write_report(results, out_dir)
    return results


def load_model_and_data(cfg: DictConfig):
    """Build the datamodule and the PIGVAE module from the composed config, load the checkpoint."""
    dm = instantiate(cfg.data)
    dm.setup(stage="fit" if cfg.split == "val" else "test")
    dataloader = dm.val_dataloader() if cfg.split == "val" else dm.test_dataloader()
    # Keep the raw pre-PCA input on inspection batches so we can measure error vs the input x.
    # Set before iterating so it propagates to dataloader workers. Training is unaffected.
    collate_fn = getattr(dataloader, "collate_fn", None)
    if collate_fn is not None and hasattr(collate_fn, "keep_input"):
        collate_fn.keep_input = True

    model = instantiate(cfg.model)

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    epoch = ckpt.get("epoch", 0)
    tau = model.temperature_scheduler(epoch)
    print(f"[info] Loaded checkpoint — epoch {epoch}, tau={tau:.4f}")

    model.eval()
    # pca_layer (built by the collator during val_dataloader) enables image-space inverse-PCA metrics
    return model, dataloader, tau, getattr(dm, "pca_layer", None)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="diagnostics/model")
def main(cfg: DictConfig) -> None:
    model, dataloader, tau, pca_layer = load_model_and_data(cfg)
    if cfg.tau is not None:
        tau = cfg.tau
        print(f"[info] tau overridden to {tau}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if cfg.inspect:
        out_dir = cfg.out_dir or str(
            Path(cfg.paths.log_dir) / "diagnostics" / Path(cfg.ckpt_path).stem
        )
        meta = {
            "ckpt": cfg.ckpt_path,
            "split": cfg.split,
            "tau": tau,
            "run": Path(cfg.ckpt_path).parent.name,
        }
        inspect_batch = next(iter(dataloader)).to(device)
        graph_ae = model.graph_ae if hasattr(model, "graph_ae") else model
        inspect_model(graph_ae, inspect_batch, out_dir, meta, pca_layer)
        print(f"[inspect] wrote artifacts to {out_dir}")

    n_run = 0
    for batch_i, batch in enumerate(dataloader):
        if cfg.batch_idx is not None and batch_i != cfg.batch_idx:
            continue

        batch = batch.to(device)
        results = run_diagnostics(model, batch, tau)
        print_report(results, batch_idx=batch_i)

        n_run += 1
        if cfg.batch_idx is not None or n_run >= cfg.num_batches:
            break

    if n_run == 0:
        print(f"[ERROR] batch_idx={cfg.batch_idx} was not reached.")
        sys.exit(1)


if __name__ == "__main__":
    main()
