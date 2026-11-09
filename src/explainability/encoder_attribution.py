"""Integrated-Gradients attribution over the PIGVAE encoder input.

Representation-level explainability: attribute a scalar target read from ``z_global``
(the CLS graph embedding) back to the encoder's per-node input features, then optionally
map the attribution from PCA space to the original marker channels.

The encoder is permutation-equivariant and ``z_global`` is (by construction) D4-invariant,
so these attributions describe *what the representation encodes*, per grid node. The
decision-level (ABMIL) attribution lives in ``abmil_attribution`` and composes on top.

Captum's Integrated Gradients is used deliberately: it needs only forward+backward through
the model (no hand-written LRP rules for the custom RMSNorm/SwiGLU/qk-norm layers), which is
what makes it robust on this bespoke graph transformer.
"""

from __future__ import annotations

import torch

from src.data.components.graphs_datamodules import DenseGraphBatch


def _encode_zglobal(gae: torch.nn.Module, node_features: torch.Tensor) -> torch.Tensor:
    """Differentiable map: encoder input ``node_features`` [B, N, F] -> ``z_global`` [B, D].

    Builds the full-valid mask the encoder expects (16x16 grid has no padding) and calls
    ``GraphAE.encode`` deterministically (``sample=False`` -> z = mu, dropout off in eval).
    """
    batch_size, num_nodes, _ = node_features.shape
    mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=node_features.device)
    graph = DenseGraphBatch(
        node_features=node_features,
        edge_features=torch.empty(0, device=node_features.device),
        mask=mask,
    )
    _z_nodes, z_global, *_ = gae.encode(graph, sample=False)
    return z_global


def encoder_ig(
    gae: torch.nn.Module,
    node_features: torch.Tensor,
    target: int,
    baselines: torch.Tensor | None = None,
    n_steps: int = 50,
    return_convergence_delta: bool = False,
):
    """Integrated Gradients of ``z_global[:, target]`` w.r.t. the encoder input.

    Args:
        gae: a ``GraphAE`` (``pl_module.graph_ae``); put in eval, params may be frozen.
        node_features: encoder input, shape ``[B, N, F]`` (PCA features per grid node).
        target: which ``z_global`` dimension to explain (0 <= target < D).
        baselines: reference input ``[B, N, F]``; defaults to zeros (the natural PCA origin).
        n_steps: IG Riemann steps.
        return_convergence_delta: also return the IG completeness gap (attr sum vs Δoutput).

    Returns:
        attributions ``[B, N, F]`` (float), or ``(attributions, delta)`` when requested.
    """
    from captum.attr import IntegratedGradients

    gae.eval()
    node_features = node_features.detach().clone().float().requires_grad_(True)
    if baselines is None:
        baselines = torch.zeros_like(node_features)

    ig = IntegratedGradients(lambda nf: _encode_zglobal(gae, nf))
    out = ig.attribute(
        node_features,
        baselines=baselines,
        target=target,
        n_steps=n_steps,
        return_convergence_delta=return_convergence_delta,
    )
    if return_convergence_delta:
        attributions, delta = out
        return attributions.detach(), delta.detach()
    return out.detach()


def pca_to_markers(attr_pca: torch.Tensor, pca_components: torch.Tensor) -> torch.Tensor:
    """Map a per-node attribution from PCA space to the original marker channels.

    PCA is a linear transform ``x_pca = (x - mean) @ Vᵀ`` with ``V = pca_components``
    (shape ``[F, C]``: F components, C markers). By the chain rule the gradient/attribution
    w.r.t. the C markers is ``attr_pca @ V``. This is exact (PCA is linear), so it reuses the
    fitted ``pca_model_*.pkl`` (``sklearn`` ``.components_``) with no approximation.

    Args:
        attr_pca: ``[B, N, F]`` attribution in PCA space.
        pca_components: ``[F, C]`` PCA loadings (``ipca.components_``).

    Returns:
        ``[B, N, C]`` attribution in marker space.
    """
    pca_components = pca_components.to(attr_pca.dtype).to(attr_pca.device)
    return attr_pca @ pca_components
