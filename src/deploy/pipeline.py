"""Raw cords IMC image -> clinical prediction with per-crop explanations, as plain PyTorch.

The chain reproduces the thesis pipeline::

    raw [43, H, W] -> preprocess -> 128 px crops -> ImmuVis [n, 768, 16, 16]
      -> PCA-128 per node [n, 256, 128] -> PIGVAE z_global [n, 512]
      -> fold z-score -> gated ABMIL -> logit, attention; Integrated Gradients per crop

Everything is loaded from a self-contained bundle directory written by
``scripts/deploy/export_bundle.py``; no Hydra config, raid path or W&B is read. This module has
no Modal dependency, so it runs and is tested locally (``src/deploy/modal/app.py`` serves it).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.imc_raw import CROP, crops, preprocess
from src.deploy.preview import DEFAULT_PREVIEW_MARKERS, render_preview, resolve_markers

N_MARKERS = 43  # cords panel
MAX_SIDE = 4096  # pixels; larger uploads are rejected
GRID = 16  # ImmuVis latent grid per crop (16 x 16 nodes)


class InvalidImageError(ValueError):
    """The uploaded image cannot be processed (shape, size or values)."""


def validate_image(
    img: np.ndarray, n_markers: int = N_MARKERS, max_side: int = MAX_SIDE
) -> np.ndarray:
    """Check a raw IMC image and return it as float32 ``[n_markers, H, W]``.

    A channels-last ``[H, W, n_markers]`` array is transposed. Raises :class:`InvalidImageError`
    on a wrong rank or channel count, a side shorter than one crop or longer than ``max_side``,
    or non-finite values.
    """
    if img.ndim != 3:
        raise InvalidImageError(f"expected a 3-D image [{n_markers}, H, W], got shape {img.shape}")
    if img.shape[0] != n_markers and img.shape[-1] == n_markers:
        img = np.moveaxis(img, -1, 0)
    if img.shape[0] != n_markers:
        raise InvalidImageError(
            f"expected {n_markers} channels (cords panel), got shape {img.shape}"
        )
    _, height, width = img.shape
    if min(height, width) < CROP:
        raise InvalidImageError(f"image {height}x{width} is smaller than one {CROP} px crop")
    if max(height, width) > max_side:
        raise InvalidImageError(f"image {height}x{width} exceeds {max_side} px per side")
    img = img.astype(np.float32, copy=False)
    if not np.isfinite(img).all():
        raise InvalidImageError("image contains NaN or infinite values")
    return img


def project_nodes(
    latent: torch.Tensor, components: torch.Tensor, mean: torch.Tensor
) -> torch.Tensor:
    """ImmuVis maps ``[B, 768, 16, 16]`` -> PCA node features ``[B, 256, 128]`` (row-major nodes)."""
    b, c = latent.shape[:2]
    nodes = latent.reshape(b, c, -1).transpose(1, 2)
    return (nodes - mean) @ components.T


@dataclass
class AbmilHead:
    """A trained gated-ABMIL head with the z-score statistics of its training fold."""

    model: torch.nn.Module
    mu: torch.Tensor  # [D]
    sd: torch.Tensor  # [D]
    classes: list[str]
    label: str
    fold: int
    fold_val_auroc: float


@dataclass
class Embedding:
    """Per-crop PIGVAE embeddings of one image."""

    coords: list[tuple[int, int]]  # top-left (row, col) of each crop, row-major
    z_global: torch.Tensor  # [n, D] on CPU
    height: int
    width: int
    timings_ms: dict[str, float] = field(default_factory=dict)
    preprocessed: np.ndarray | None = None  # [43, H, W] in [0, 1], for the preview


class IMCPipeline:
    """The full chain; every stage is injected so tests can use small stand-ins.

    Args:
        immuvis: preprocessed crops ``[B, 43, 128, 128]`` -> ImmuVis maps ``[B, 768, 16, 16]``.
        pca_components: ``[128, 768]``; ``pca_mean``: ``[768]``.
        gae: the PIGVAE ``GraphAE`` (``encode(batch, sample=False)[1]`` is ``z_global``).
        head: the ABMIL head with its fold statistics.
        markers: cords marker names in channel order.
        manifest: bundle provenance, reported by :meth:`info`.
    """

    def __init__(
        self,
        immuvis: Callable[[torch.Tensor], torch.Tensor],
        pca_components: torch.Tensor,
        pca_mean: torch.Tensor,
        gae: torch.nn.Module,
        head: AbmilHead,
        markers: list[str],
        manifest: dict[str, Any] | None = None,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.immuvis = immuvis
        self.device = device
        self.components = pca_components.float().to(device)
        self.pca_mean = pca_mean.float().to(device)
        self.gae = gae.to(device).eval()
        self.head = head
        self.head.model.to(device).eval()
        self.mu, self.sd = head.mu.float().to(device), head.sd.float().to(device)
        self.markers = markers
        self.manifest = manifest or {}
        self.batch_size = batch_size

    # ---- loading ------------------------------------------------------------------------------
    @classmethod
    def from_bundle(
        cls, bundle_dir: str | Path, device: str = "cpu", code_root: str | Path | None = None
    ) -> IMCPipeline:
        """Load every stage from a bundle written by ``scripts/deploy/export_bundle.py``."""
        from omegaconf import OmegaConf

        from src.downstream.abmil.model import GatedABMIL
        from src.models import immuvis as immuvis_lib
        from src.models.components.modules import GraphAE

        bundle = Path(bundle_dir)
        manifest = json.loads((bundle / "manifest.json").read_text())
        markers = json.loads((bundle / "markers.json").read_text())

        kwargs = {} if code_root is None else {"code_root": code_root}
        encoder, (missing, unexpected) = immuvis_lib.load_immuvis(
            bundle / "immuvis.pth", bundle / "immuvis.yaml", device, **kwargs
        )
        if missing or unexpected:
            raise RuntimeError(f"ImmuVis load: missing {missing[:3]}, unexpected {unexpected[:3]}")
        encoder.requires_grad_(False)
        channel_ids = torch.tensor(markers["channel_ids"], dtype=torch.long, device=device)

        def run_immuvis(x: torch.Tensor) -> torch.Tensor:
            return immuvis_lib.encode(encoder, x, channel_ids[None].expand(x.shape[0], -1))

        pca = torch.load(bundle / "pca.pt", map_location="cpu")
        pig = torch.load(bundle / "pigvae.pt", map_location="cpu")
        gae = GraphAE(OmegaConf.create(pig["hparams"]))
        gae.load_state_dict(pig["state_dict"], strict=True)
        gae.requires_grad_(False)

        ab = torch.load(bundle / manifest["abmil"], map_location="cpu")
        abmil = GatedABMIL(**ab["hparams"])
        abmil.load_state_dict(ab["state_dict"], strict=True)
        head = AbmilHead(
            model=abmil,
            mu=ab["mu"],
            sd=ab["sd"],
            classes=list(ab["classes"]),
            label=ab["label"],
            fold=int(ab["fold"]),
            fold_val_auroc=float(ab["fold_val_auroc"]),
        )
        return cls(
            run_immuvis,
            pca["components"],
            pca["mean"],
            gae,
            head,
            markers["markers"],
            manifest,
            device,
        )

    # ---- stages -------------------------------------------------------------------------------
    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    @torch.no_grad()
    def embed(self, raw: np.ndarray) -> Embedding:
        """Raw image ``[43, H, W]`` (or channels-last) -> per-crop ``z_global``."""
        from src.data.components.graphs_datamodules import DenseGraphBatch

        img = validate_image(raw)
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        pre = preprocess(img)
        tiles = crops(pre)
        timings["preprocess"] = (time.perf_counter() - t0) * 1e3

        coords = [ij for ij, _ in tiles]
        stack = torch.from_numpy(np.stack([t for _, t in tiles])).float()
        t_immuvis = t_pigvae = 0.0
        out = []
        for s in range(0, len(stack), self.batch_size):
            x = stack[s : s + self.batch_size].to(self.device)
            t0 = time.perf_counter()
            latent = self.immuvis(x)
            self._sync()
            t_immuvis += time.perf_counter() - t0

            t0 = time.perf_counter()
            nodes = project_nodes(latent, self.components, self.pca_mean)
            mask = torch.ones(nodes.shape[:2], dtype=torch.bool, device=self.device)
            batch = DenseGraphBatch(
                node_features=nodes, edge_features=torch.empty(0, device=self.device), mask=mask
            )
            out.append(self.gae.encode(batch, sample=False)[1].float())
            self._sync()
            t_pigvae += time.perf_counter() - t0
        timings["immuvis"], timings["pigvae"] = t_immuvis * 1e3, t_pigvae * 1e3
        _, height, width = img.shape
        return Embedding(coords, torch.cat(out).cpu(), height, width, timings, pre)

    def predict(
        self,
        raw: np.ndarray,
        ig_steps: int = 50,
        include_embeddings: bool = False,
        include_preview: bool = False,
        preview_markers: Sequence[str] = DEFAULT_PREVIEW_MARKERS,
    ) -> dict[str, Any]:
        """Prediction, attention and Integrated Gradients per crop, as a JSON-ready dict.

        ``ig_importance`` is each crop's signed Integrated-Gradients contribution: for a binary
        (single-logit) head to that logit, so positive pushes toward ``classes[1]``; for a
        multi-class head to the logit of the predicted class. ``ig_steps=0`` skips IG.
        ``include_preview`` adds an RGB PNG of ``preview_markers`` (see :mod:`.preview`).
        """
        from src.explainability.abmil_attribution import abmil_attention, abmil_ig, crop_importance

        if include_preview:  # fail on bad marker names before the expensive part
            resolve_markers(preview_markers, self.markers)
        emb = self.embed(raw)
        timings = dict(emb.timings_ms)
        model = self.head.model
        bag = ((emb.z_global.to(self.device) - self.mu) / self.sd)[None]  # [1, n, D]

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(bag)[0][0]
            attention = abmil_attention(model, bag)[0, :, 0]
        classes = self.head.classes
        if logits.numel() == 1:  # binary single-logit head
            p1 = torch.sigmoid(logits[0]).item()
            probs = [1.0 - p1, p1]
            target = 0
        else:
            probs = torch.softmax(logits, dim=0).tolist()
            target = int(np.argmax(probs))
        pred = int(np.argmax(probs))
        self._sync()
        timings["abmil"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        if ig_steps > 0:
            importance = crop_importance(abmil_ig(model, bag, target=target, n_steps=ig_steps))[0]
            importance = importance.cpu().tolist()
        else:
            importance = [None] * len(emb.coords)
        timings["ig"] = (time.perf_counter() - t0) * 1e3

        z = emb.z_global.tolist() if include_embeddings else [None] * len(emb.coords)
        out: dict[str, Any] = {
            "label": self.head.label,
            "prediction": classes[pred],
            "probabilities": dict(zip(classes, probs, strict=True)),
            "logit": logits.cpu().tolist() if logits.numel() > 1 else logits.item(),
            "image": {
                "height": emb.height,
                "width": emb.width,
                "tile_size": CROP,
                "n_crops": len(emb.coords),
            },
            "crops": [
                {
                    "row": r,
                    "col": c,
                    "attention": a,
                    "ig_importance": g,
                    "z_global": zz,
                }
                for (r, c), a, g, zz in zip(
                    emb.coords, attention.cpu().tolist(), importance, z, strict=True
                )
            ],
            "model": self._model_info(),
            "timing_ms": {k: round(v, 1) for k, v in timings.items()},
        }
        if include_preview:
            t0 = time.perf_counter()
            out["preview"] = render_preview(emb.preprocessed, preview_markers, self.markers)
            out["timing_ms"]["preview"] = round((time.perf_counter() - t0) * 1e3, 1)
        return out

    def embed_json(self, raw: np.ndarray) -> dict[str, Any]:
        """``/embed`` response: image size and per-crop ``z_global``."""
        emb = self.embed(raw)
        return {
            "image": {
                "height": emb.height,
                "width": emb.width,
                "tile_size": CROP,
                "n_crops": len(emb.coords),
            },
            "crops": [
                {"row": r, "col": c, "z_global": z}
                for (r, c), z in zip(emb.coords, emb.z_global.tolist(), strict=True)
            ],
            "model": self._model_info(),
            "timing_ms": {k: round(v, 1) for k, v in emb.timings_ms.items()},
        }

    def _model_info(self) -> dict[str, Any]:
        return {
            "label": self.head.label,
            "fold": self.head.fold,
            "fold_val_auroc": self.head.fold_val_auroc,
            "bundle": self.manifest.get("bundle_id"),
        }

    def info(self) -> dict[str, Any]:
        """Label, classes, fold, the expected marker order and the bundle manifest."""
        return {
            **self._model_info(),
            "classes": self.head.classes,
            "markers": self.markers,
            "n_markers": len(self.markers),
            "tile_size": CROP,
            "max_side": MAX_SIDE,
            "device": self.device,
            "manifest": self.manifest,
        }
