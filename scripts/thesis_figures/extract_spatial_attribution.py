"""Spatial attribution examples for Sec. 6.3 (szary, GPU): reference run, label DX.name.

For one correctly classified slide per class (out-of-fold head):
  - per-crop ABMIL decision attribution: Integrated Gradients on the bag (zero baseline in the
    train-fold z-scored space, 50 steps), summed over features, oriented to the predicted class;
    the head-averaged attention; tissue density / empty flag per crop; the preprocessed slide;
  - end-to-end raw-image attribution for the strongest real-tissue crop: Integrated Gradients on the
    preprocessed 43x128x128 crop through ImmuVis -> PCA-128 -> PIGVAE -> z_global -> train-fold
    z-score -> ABMIL logit, the bag's other crops held fixed (zero baseline).
Checks: ImmuVis(crop) vs the stored h5 embedding (also used to pick the TIFF source and coordinate
order), and the chain logit vs the head logit. -> ``figures/_cache/spatial_v3/<class>_<image>.npz``

Usage:
    uv run python scripts/thesis_figures/extract_spatial_attribution.py
"""

import argparse
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

import numpy as np
import tifffile
import torch
from captum.attr import IntegratedGradients

from figlib import abmil, imc, paths
from figlib.chain import ReferenceChain

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = imc.CROP
BAG_IG_STEPS = 50
RAW_IG_STEPS = (64, 128, 256)  # completeness is reported for each; the last one is saved


def pick_examples(cohort: abmil.Cohort, oof: np.ndarray) -> list[int]:
    """Per class, the most confidently correct slide with 25-42 crops and <= 20% empty crops."""
    n_crops = np.array([len(b) for b in cohort.bags])
    empty_frac = np.array([cohort.empty(i).mean() for i in range(len(cohort.bags))])
    picks = []
    for cls in [0, 1]:
        ok = (
            (cohort.y == cls)
            & ((oof > 0) == (cls == 1))
            & (n_crops >= 25)
            & (n_crops <= 42)
            & (empty_frac <= 0.2)
        )
        idx = np.where(ok)[0]
        picks.append(int(idx[np.argmax(np.abs(oof[idx]))]))
    print(
        "classes",
        cohort.classes,
        "picked",
        [
            (cohort.names[i], cohort.classes[cohort.y[i]], round(oof[i], 2), n_crops[i])
            for i in picks
        ],
        flush=True,
    )
    return picks


def matching_crop(chain: ReferenceChain, image: str, a0: int, a1: int, stored: torch.Tensor):
    """Find the TIFF source and coordinate order whose ImmuVis map best reproduces the stored one."""
    sources = {"full": paths.TIFF_FULL / image}
    for split in ("train", "test"):
        candidate = paths.TIFF_SPLIT / split / "cords" / "imgs" / image
        if candidate.exists():
            sources["split_" + split] = candidate
    best = None
    for name, path in sources.items():
        slide = imc.preprocess(tifffile.imread(path)).astype(np.float32)
        for orient, (r0, c0) in {"rc": (a0, a1), "cr": (a1, a0)}.items():
            if r0 + P > slide.shape[1] or c0 + P > slide.shape[2]:
                continue
            crop = torch.from_numpy(slide[:, r0 : r0 + P, c0 : c0 + P].copy())[None].to(DEVICE)
            with torch.no_grad():
                latent = chain.encode_crops(crop)[1]
            corr = torch.corrcoef(torch.stack([latent[0].flatten().float(), stored.flatten()]))[
                0, 1
            ].item()
            max_abs = (latent[0].float() - stored).abs().max().item()
            print(
                f"      source={name:11s} orient={orient} shape={slide.shape} corr={corr:.5f} maxabs={max_abs:.4g}",
                flush=True,
            )
            if best is None or corr > best[0]:
                best = (corr, name, orient, slide, r0, c0)
    return best


def explain_slide(tag: str, i: int, cohort, models, chain: ReferenceChain, out_dir: Path) -> None:
    f = cohort.fold_of[i]
    head = models.heads[f]
    mu, sd = (torch.from_numpy(s).float().to(DEVICE) for s in models.stats[f])
    image = cohort.names[i]
    bag = models.zscored_bag(cohort, i, DEVICE)
    with torch.no_grad():
        logit, att = abmil.gated_attention(head, bag)
    sign = 1.0 if logit.item() > 0 else -1.0
    crop_ig = (
        IntegratedGradients(lambda b: head(b)[0][:, 0] * sign)
        .attribute(bag, baselines=torch.zeros_like(bag), n_steps=BAG_IG_STEPS)
        .sum(-1)[0]
        .cpu()
        .numpy()
    )
    coords = cohort.coords(i)
    density = cohort.density(i)
    empty = density["empty"].values.astype(bool)
    k = int(np.argmax(np.where(~empty, crop_ig, -np.inf)))  # strongest real-tissue crop
    a0, a1 = coords[k]

    stored = chain.stored_embedding(image, a0, a1)
    corr, source, orient, slide, r0, c0 = matching_crop(chain, image, a0, a1, stored)
    crop = torch.from_numpy(slide[:, r0 : r0 + P, c0 : c0 + P].copy())[None].to(DEVICE)
    print(f"      using source={source} orient={orient} corr={corr:.5f}", flush=True)

    def chain_logit(x):
        """Oriented ABMIL logit with crop k replaced by the re-encoded z_global of x."""
        z = (chain.encode_crops(x)[0] - mu) / sd
        b = bag.expand(x.shape[0], -1, -1).clone()
        b[:, k] = z
        return head(b)[0][:, 0] * sign

    with torch.no_grad():
        check_logit = (chain_logit(crop).item() * sign, logit.item())
        delta = chain_logit(crop).item() - chain_logit(torch.zeros_like(crop)).item()
    print(
        f"[{tag}] {image} fold {f} logit {logit.item():+.3f} | crop {k} at ({r0},{c0}) | corr(ImmuVis, stored)="
        f"{corr:.5f} | chain logit {check_logit[0]:+.4f} vs head {check_logit[1]:+.4f}",
        flush=True,
    )
    completeness = {}
    for steps in RAW_IG_STEPS:
        pixel_attr = (
            IntegratedGradients(chain_logit)
            .attribute(
                crop, baselines=torch.zeros_like(crop), n_steps=steps, internal_batch_size=8
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        completeness[steps] = float(pixel_attr.sum())
        print(
            f"      IG n_steps={steps:3d}: sum(attr) {completeness[steps]:+.4f} vs f(x)-f(0) {delta:+.4f} "
            f"(rel err {abs(completeness[steps] - delta) / max(abs(delta), 1e-8):.2f})",
            flush=True,
        )

    np.savez_compressed(
        out_dir / f"{tag}_{image[:-5]}.npz",
        img=image,
        label=cohort.classes[cohort.y[i]],
        logit=logit.item(),
        fold=f,
        coords=coords,
        ig=crop_ig * 1.0,
        att=att.mean(1)[0].cpu().numpy(),
        density=density["density"].values,
        empty=empty,
        decisive=k,
        slide=slide.astype(np.float16),
        crop_attr=pixel_attr.astype(np.float32),
        markers=np.array(chain.markers),
        check_corr=corr,
        check_logit=np.array(check_logit),
        src=source,
        orient=orient,
        crop_rc=np.array([r0, c0]),
        delta=delta,
        completeness=np.array([[s, v] for s, v in completeness.items()]),
    )


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cohort = abmil.load_cohort("DX.name")
    models = abmil.load_fold_models(cohort, DEVICE)
    picks = pick_examples(cohort, abmil.oof_logits(cohort, models, DEVICE))
    chain = ReferenceChain(DEVICE, with_immuvis=True)
    for tag, i in zip(["adeno", "squam"], picks, strict=True):
        explain_slide(tag, i, cohort, models, chain, out)
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(paths.CACHE / "spatial_v3"))
    main(ap.parse_args().out_dir)
