"""
Diagnostic inference script for PIGVAE.

Loads a checkpoint, runs a few validation batches, and prints a detailed report
covering every subsystem that could be dead or collapsed:
  - Latent space (mu, std, active dims, KLD per dim)
  - Node feature diversity across D4 augmentations (position-level, not mean-pooled)
  - Raw permuter scores before Sinkhorn (are they identical for all 8 views?)
  - Permuter predictions and confidence per augmentation slot
  - Activation stats in encoder transformer, permuter transformer, decoder

Usage:
    python scripts/diagnose_model.py \\
        --ckpt /path/to/checkpoint.ckpt \\
        [--num-batches 3] \\
        [--batch-idx 5]        # pick a specific batch instead
        [--split val]          # val (default) or test
        [--tau 0.5]            # temperature (default: from checkpoint)
        [--data-dir /path/to/data]
"""

import argparse
import operator
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
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


def _stat(t: torch.Tensor, label: str = "") -> dict:
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

    # Register activation hooks
    for name, module in [
        ("encoder/graph_transformer", ae.encoder.graph_transformer),
        ("permuter/spectral_emb",     ae.permuter.spectral_embeddings),
        ("permuter/graph_transformer", ae.permuter.graph_transformer),
        ("permuter/scoring_fc",        ae.permuter.scoring_fc),   # raw scores before Sinkhorn
        ("decoder/deconv2",            ae.decoder.deconv2),
        ("decoder/fc_out",             ae.decoder.fc_out),
    ]:
        keep = (name == "permuter/scoring_fc")  # keep full tensor for score analysis
        hooks.append(module.register_forward_hook(make_activation_hook(act_store, name, keep_tensor=keep)))

    with torch.no_grad():
        graph_emb, graph_pred, soft_probs, perm, mu, logvar, node_features = model(
            graph=batch, training=False, tau=tau
        )

    for h in hooks:
        h.remove()

    total_B = node_features.shape[0]
    B = total_B // NUM_VIEWS
    N = node_features.shape[1]
    D = node_features.shape[2]

    # ── 1. Latent space ────────────────────────────────────────────────────
    std = (0.5 * logvar).exp()                                 # [8B, emb_dim]
    kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - std.pow(2))  # [8B, emb_dim]
    kld_per_dim_mean = kld_per_dim.mean(dim=0)                 # [emb_dim]
    active = (kld_per_dim_mean > 0.1).sum().item()

    latent_diag = {
        "mu_mean":          round(mu.mean().item(), 5),
        "mu_std":           round(mu.std().item(), 5),
        "std_mean":         round(std.mean().item(), 5),
        "std_per_dim_min":  round(std.mean(0).min().item(), 5),
        "std_per_dim_max":  round(std.mean(0).max().item(), 5),
        "active_dims":      active,
        "total_dims":       mu.shape[-1],
        "kld_per_dim_median": round(kld_per_dim_mean.median().item(), 5),
        "kld_per_dim_p95":  round(kld_per_dim_mean.quantile(0.95).item(), 5),
        "kld_per_dim_max":  round(kld_per_dim_mean.max().item(), 5),
        "total_kld_mean":   round(kld_per_dim.mean().item(), 5),
    }

    # ── 2. Node feature diversity ─────────────────────────────────────────
    views = node_features.view(NUM_VIEWS, B, N, D)     # [8, B, N, D]
    views_norm = F.normalize(views, dim=-1)

    # Position-level comparison (the correct metric)
    sim_pos = torch.einsum("vbnd,ubnd->uvb", views_norm, views_norm) / N  # [8, 8, B]
    off_diag = ~torch.eye(NUM_VIEWS, dtype=torch.bool)

    # Mean-pooled comparison (to confirm it's always ~1 — sanity check of old metric)
    views_mean = views.mean(dim=2)                              # [8, B, D]
    views_mean_norm = F.normalize(views_mean, dim=-1)
    sim_mean = torch.einsum("vbd,ubd->uvb", views_mean_norm, views_mean_norm) / 1  # [8, 8, B]

    # Per-augmentation pair similarity (averaged over images in batch)
    sim_pair = sim_pos.mean(dim=-1)  # [8, 8]

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
    probs = soft_probs.view(NUM_VIEWS, B, NUM_VIEWS)  # [8, B, 8]
    pred_classes = probs.argmax(dim=-1)                # [8, B]
    confidence = probs.max(dim=-1).values              # [8, B]
    per_row_H = -(probs * (probs + 1e-8).log()).sum(-1)  # [8, B]

    # Check if all 8 augmented views of the same image get identical raw scores
    raw_scores = act_store.get("permuter/scoring_fc__tensor")  # [8B, 8]
    score_diag = {}
    if raw_scores is not None:
        scores_by_view = raw_scores.view(NUM_VIEWS, B, NUM_VIEWS)  # [8, B, 8]
        # Max difference between any two views for the same image
        max_score_diff = (
            scores_by_view.unsqueeze(0) - scores_by_view.unsqueeze(1)
        ).abs().max().item()
        mean_score_diff = (
            scores_by_view.unsqueeze(0) - scores_by_view.unsqueeze(1)
        ).abs().mean().item()
        # Literal score vectors for image 0 across all 8 augmentations.
        # If identical → encoder produces identical features for all augmentations.
        img0_scores = scores_by_view[:, 0, :]   # [8, 8]: aug × class
        img0_diffs  = img0_scores - img0_scores[0:1, :]  # diff relative to r0_f

        score_diag = {
            "raw_scores_mean":         round(raw_scores.mean().item(), 5),
            "raw_scores_std":          round(raw_scores.std().item(), 5),
            "raw_scores_min":          round(raw_scores.min().item(), 5),
            "raw_scores_max":          round(raw_scores.max().item(), 5),
            "max_diff_across_views":   round(max_score_diff, 6),  # 0 = all views identical
            "mean_diff_across_views":  round(mean_score_diff, 6),
            "IMAGE_0_score_vectors (8 classes each)": {
                AUG_NAMES[v]: [round(x, 5) for x in img0_scores[v].tolist()]
                for v in range(NUM_VIEWS)
            },
            "IMAGE_0_diff_vs_r0_f (should be nonzero)": {
                AUG_NAMES[v]: [round(x, 7) for x in img0_diffs[v].tolist()]
                for v in range(NUM_VIEWS)
            },
        }

    perm_diag = {
        "mean_confidence":           round(confidence.mean().item(), 5),
        "uniform_confidence_is":     round(1.0 / NUM_VIEWS, 5),
        "per_row_entropy_mean":      round(per_row_H.mean().item(), 5),
        "max_possible_entropy":      round(np.log(NUM_VIEWS), 5),
        "all_views_same_class":      bool((pred_classes == pred_classes[0:1]).all().item()),
        "is_bijective":              bool(len({pred_classes[:, 0].tolist()[v] for v in range(NUM_VIEWS)}) == NUM_VIEWS),
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

    # ── 4. Activation stats (from hooks) ─────────────────────────────────
    activation_diag = {k: v for k, v in act_store.items() if not k.endswith("__tensor")}

    # ── 5. Equivariance proof ─────────────────────────────────────────────
    # Batch layout: aug_idx=0 → r0_f (Y-flip), aug_idx=1 → r0_nf (identity).
    # For image b, flat indices are: r0_f at b, r0_nf at b+B.
    # If the encoder is permutation-equivariant: encoder(flip(x)) = flip(encoder(x))
    # So: node_features[b] should equal flip_perm(node_features[b+B])
    # where flip_perm is the Y-axis flip permutation of the 6×6 grid.
    grid = 6
    n_nodes = grid * grid
    # Y-axis flip: node (row, col) → (row, grid-1-col)
    flip_perm = [row * grid + (grid - 1 - col)
                 for row in range(grid) for col in range(grid)]  # length 36

    h_identity = node_features[B].cpu()           # r0_nf of image 0: [N, D]
    h_flip     = node_features[0].cpu()            # r0_f  of image 0: [N, D]

    # Apply the flip permutation to the identity features
    h_identity_permuted = h_identity[flip_perm, :]  # [N, D]

    raw_diff    = (h_flip - h_identity).abs()           # diff WITHOUT applying permutation
    perm_diff   = (h_flip - h_identity_permuted).abs()  # diff AFTER applying permutation

    # Per-node cosine similarity before and after applying permutation
    cos_before = F.cosine_similarity(h_flip, h_identity, dim=-1).mean().item()
    cos_after  = F.cosine_similarity(h_flip, h_identity_permuted, dim=-1).mean().item()

    equivariance_diag = {
        "claim": "encoder(flip(x)) == flip(encoder(x)) iff encoder is permutation-equivariant",
        "h_identity shape": list(h_identity.shape),
        "h_flip     shape": list(h_flip.shape),
        "diff WITHOUT permutation": {
            "max":  round(raw_diff.max().item(), 6),
            "mean": round(raw_diff.mean().item(), 6),
            "cos_sim": round(cos_before, 6),
            "interpretation": "should be large if augmentation changed content",
        },
        "diff AFTER applying flip_perm to identity": {
            "max":  round(perm_diff.max().item(), 6),
            "mean": round(perm_diff.mean().item(), 6),
            "cos_sim": round(cos_after, 6),
            "interpretation": "should be ~0 / cos~1 if encoder is equivariant",
        },
        "verdict": (
            "EQUIVARIANT (diff<1e-3 after perm)" if perm_diff.max().item() < 1e-3
            else "NOT perfectly equivariant or content collapsed"
        ),
    }

    # ── 6. Within-augmentation inter-node similarity ─────────────────────────
    # Is cos(h_i, h_j) ~ 1 for different positions i≠j in the SAME augmentation?
    # This is NOT guaranteed by equivariance — it would mean the encoder outputs
    # the same vector at every position regardless of which cell is there.
    h_id_all = node_features[B].cpu()          # r0_nf (identity), all 36 nodes: [36, 256]
    h_id_norm = F.normalize(h_id_all, dim=-1)  # [36, 256]
    sim_within = torch.mm(h_id_norm, h_id_norm.t())  # [36, 36]
    off_diag_36 = ~torch.eye(36, dtype=torch.bool)
    inter_node_cos = sim_within[off_diag_36]

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

    # ── 7. Raw tensor inspection: same node, two augmentations ──────────────
    # Show the actual 256-dim vectors for a few node positions so you can
    # visually confirm whether they are nearly identical or truly different.
    # Image 0: r0_nf (identity) at flat index B, r0_f (flip) at flat index 0.
    node_inspection = {}
    for node_idx in [0, 5, 17, 35]:
        v_identity = h_identity[node_idx]  # [256]
        v_flip     = h_flip[node_idx]      # [256] — different physical cell here
        cos = F.cosine_similarity(v_identity.unsqueeze(0), v_flip.unsqueeze(0)).item()
        l2  = (v_identity - v_flip).norm().item()
        node_inspection[f"node_{node_idx}"] = {
            "r0_nf (identity) first 16 dims": [round(x, 4) for x in v_identity[:16].tolist()],
            "r0_f  (flip)     first 16 dims": [round(x, 4) for x in v_flip[:16].tolist()],
            "element-wise diff first 16 dims": [round(x, 4) for x in (v_flip - v_identity)[:16].tolist()],
            "cosine_similarity": round(cos, 6),
            "L2_norm_of_diff":   round(l2, 6),
            "L2_norm_of_vector": round(v_identity.norm().item(), 4),
        }

    return {
        "latent": latent_diag,
        "node_features": node_diag,
        "permuter": perm_diag,
        "activations": activation_diag,
        "equivariance_proof": equivariance_diag,
        "within_aug_inter_node_similarity": within_aug_diag,
        "node_inspection (image 0, r0_nf vs r0_f)": node_inspection,
    }


# ── pretty printer ────────────────────────────────────────────────────────────

def _print_section(title: str, data: dict, indent: int = 0):
    pad = "  " * indent
    print(f"\n{pad}{'─' * (60 - 2*indent)}")
    print(f"{pad}  {title}")
    print(f"{pad}{'─' * (60 - 2*indent)}")
    _print_dict(data, indent + 1)


def _print_dict(d: dict, indent: int = 0):
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            _print_dict(v, indent + 1)
        else:
            print(f"{pad}{k}: {v}")


def print_report(results: dict, batch_idx: int):
    print(f"\n{'═' * 62}")
    print(f"  PIGVAE DIAGNOSTIC REPORT — batch {batch_idx}")
    print(f"{'═' * 62}")

    _print_section("LATENT SPACE", results["latent"])
    _print_section("NODE FEATURE DIVERSITY (across D4 augmentations)", results["node_features"])
    _print_section("PERMUTER", results["permuter"])
    _print_section("ACTIVATION STATS (per module)", results["activations"])
    _print_section("EQUIVARIANCE PROOF (image 0: r0_nf vs r0_f)", results["equivariance_proof"])
    _print_section("WITHIN-AUG INTER-NODE SIMILARITY (is cos(h_i, h_j) ~ 1 for i≠j?)",
                   results["within_aug_inter_node_similarity"])
    _print_section("NODE INSPECTION — actual vectors at 4 positions (image 0: r0_nf vs r0_f)",
                   results["node_inspection (image 0, r0_nf vs r0_f)"])


# ── main ──────────────────────────────────────────────────────────────────────

def load_model_and_data(ckpt_path: str, data_dir: str | None, split: str, paths_name: str = "szary"):
    configs = project_root / "configs"

    # Load configs directly with OmegaConf — avoids Hydra compose null-override bug
    model_cfg = OmegaConf.load(configs / "model" / "model.yaml")
    data_cfg  = OmegaConf.load(configs / "data"  / "mnist.yaml")
    paths_cfg = OmegaConf.load(configs / "paths" / f"{paths_name}.yaml")

    # Register resolvers needed by the configs (safe to re-register)
    for name, fn in [("multiply", lambda x, y: int(x) * int(y)),
                     ("divide",   lambda x, y: int(x) // int(y))]:
        try:
            OmegaConf.register_new_resolver(name, fn)
        except Exception:
            pass  # already registered

    # Resolve data_dir override
    if data_dir:
        OmegaConf.update(paths_cfg, "data_dir", data_dir, merge=True)

    # Patch config references that point at trainer/data so interpolation works
    trainer_stub = OmegaConf.create({"max_epochs": 200, "min_epochs": 1})
    data_stub    = OmegaConf.create({"hparams": {"num_aug_per_sample": 8, "batch_size": 16}})

    # Merge into a context so cross-config ${...} interpolations resolve
    ctx = OmegaConf.create({
        "model":   model_cfg,
        "data":    data_stub,
        "trainer": trainer_stub,
        "paths":   paths_cfg,
    })
    OmegaConf.set_struct(ctx, False)
    # Re-attach model cfg so internal ${model.*} refs see the full tree
    ctx.model = model_cfg

    # Resolve data_cfg separately (it uses ${paths.data_dir})
    data_cfg_resolved = OmegaConf.merge(data_cfg, {})
    OmegaConf.update(data_cfg_resolved, "hparams.data_dir", OmegaConf.select(ctx, "paths.data_dir"))

    # Instantiate data module
    dm = instantiate(data_cfg_resolved)
    dm.setup(stage="fit" if split == "val" else "test")
    dataloader = dm.val_dataloader() if split == "val" else dm.test_dataloader()

    # Instantiate model components
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
        node_feat_diversity_weight=float(OmegaConf.select(ctx, "model.node_feat_diversity_weight", default=0.01)),
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
    return model, dataloader, tau


def main():
    parser = argparse.ArgumentParser(description="PIGVAE diagnostic inference")
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    parser.add_argument("--data-dir", default=None, help="Override data_dir (default: from config)")
    parser.add_argument("--paths", default="szary", help="Paths config name (default: szary)")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--num-batches", type=int, default=3, help="Number of batches to diagnose")
    parser.add_argument("--batch-idx", type=int, default=None, help="Run only this specific batch index")
    parser.add_argument("--tau", type=float, default=None, help="Override temperature tau")
    args = parser.parse_args()

    model, dataloader, tau = load_model_and_data(args.ckpt, args.data_dir, args.split, args.paths)
    if args.tau is not None:
        tau = args.tau
        print(f"[info] tau overridden to {tau}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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
        print(f"[ERROR] --batch-idx {args.batch_idx} was not reached. Dataloader has fewer batches?")
        sys.exit(1)


if __name__ == "__main__":
    main()
