# src/utils/inspection/reconstruction.py
from __future__ import annotations

import torch


def reconstruction_diagnostics(
    pred: torch.Tensor, true: torch.Tensor, num_views: int = 8
) -> dict:
    """Where reconstruction struggles: per-channel R²/MSE, per-position, per-view, vs magnitude.

    pred, true: [B, N, C].
    """
    pred = pred.detach().float()
    true = true.detach().float()
    b, n, c = true.shape
    se = (pred - true).pow(2)

    mse_ch = se.mean(dim=(0, 1))                              # [C]
    var_ch = true.var(dim=(0, 1), unbiased=False).clamp_min(1e-12)
    r2_ch = 1.0 - mse_ch / var_ch
    mse_pos = se.mean(dim=(0, 2))                             # [N]

    per_view = None
    if b % num_views == 0 and num_views > 0:
        bv = b // num_views
        per_view = se.view(num_views, bv, n, c).mean(dim=(1, 2, 3)).tolist()

    node_mag = true.norm(dim=-1).reshape(-1)                 # [B*N]
    node_err = se.mean(dim=-1).reshape(-1)
    edges = torch.quantile(
        node_mag, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=node_mag.device)
    )
    bins = []
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        m = (node_mag >= lo) & (node_mag <= hi) if i == 3 else (node_mag >= lo) & (node_mag < hi)
        bins.append({
            "mag_lo": lo.item(), "mag_hi": hi.item(),
            "mean_err": node_err[m].mean().item() if m.any() else 0.0,
            "count": int(m.sum().item()),
        })

    return {
        "overall_mse": se.mean().item(),
        "per_channel_mse": mse_ch.tolist(),
        "per_channel_r2": r2_ch.tolist(),
        "worst_channels": torch.topk(mse_ch, k=min(10, c)).indices.tolist(),
        "per_position_mse": mse_pos.tolist(),
        "per_view_mse": per_view,
        "error_by_magnitude": bins,
    }


@torch.no_grad()
def image_space_reconstruction(pred: torch.Tensor, target: torch.Tensor, pca_layer) -> dict:
    """Reconstruction error in the ORIGINAL feature space (after inverse-PCA).

    Both `pred` and `target` are the model's PCA-coefficient space `[B, N, n_pca]`. We
    inverse-transform each through `pca_layer.inverse` (which undoes z-scoring if it was
    applied and projects back through the PCA basis) to the original `[B, N, D_orig]`
    space, and measure error there. This is the metric that matters for the real pipeline
    (reconstruct -> inverse PCA -> image): errors are weighted by each component's actual
    contribution to the image, unlike the whitened coefficient MSE which over-weights the
    low-variance tail. The PCA truncation/clip loss is shared by both, so this isolates the
    model's error.
    """
    pred = pred.detach().float()
    target = target.detach().float()
    pca_layer = pca_layer.to(pred.device)
    img_pred = pca_layer.inverse(pred)      # [B, N, D_orig]
    img_true = pca_layer.inverse(target)
    se = (img_pred - img_true).pow(2)
    var_ch = img_true.var(dim=(0, 1), unbiased=False).clamp_min(1e-12)
    r2_ch = 1.0 - se.mean(dim=(0, 1)) / var_ch
    return {
        "n_orig_channels": int(img_true.shape[-1]),
        "image_mse": se.mean().item(),
        "image_mae": (img_pred - img_true).abs().mean().item(),
        "image_r2_mean": r2_ch.mean().item(),
        "image_r2_median": r2_ch.median().item(),
        "image_r2_min": r2_ch.min().item(),
    }
