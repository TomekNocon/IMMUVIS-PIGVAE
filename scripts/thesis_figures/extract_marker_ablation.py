"""Marker ablation (occlusion) through the full chain, Sec. 6.3 deep-dive (szary, GPU).

Raw crop (preprocessed) -> ImmuVis -> PCA-128 -> PIGVAE -> z_global -> train-fold z-score -> OOF ABMIL
head. Forward passes only. Effects are oriented to the predicted class: effect > 0 means the marker or
crop SUPPORTS the model's call.

Per slide (-> ``figures/_cache/marker_ablation/<image>.npz``):
  logit_ref       every crop re-encoded from the raw image (consistency vs the stored-bag logit)
  slide_abl [43]  marker m zeroed in ALL crops
  crop_abl [n,43] marker m zeroed in crop c only (other crops at reference)
  group_abl [G], group_crop_abl [n,G]  marker groups zeroed (all crops / one crop)
  crop_occ [n]    whole crop c zeroed (all markers)
  marker_mean [n,43]  mean preprocessed intensity per crop and marker
Cohort: the two example slides (7064 adeno, 6577 squamous, with the slide saved) + up to
``--per-class`` correctly classified OOF slides per class.

Usage:
    uv run python scripts/thesis_figures/extract_marker_ablation.py --per-class 15
"""

import argparse
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


import numpy as np
import tifffile
import torch

from figlib import abmil, imc, paths, style
from figlib.chain import ReferenceChain

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = imc.CROP
BATCH = 16
EXAMPLES = ("7064.tiff", "6577.tiff")


def select_slides(cohort: abmil.Cohort, oof: np.ndarray, per_class: int) -> dict[int, str]:
    """The two examples plus a random sample of correctly classified slides with >= 10 crops."""
    rng = np.random.default_rng(0)
    n_crops = np.array([len(b) for b in cohort.bags])
    selected = {cohort.names.index(name): "example" for name in EXAMPLES}
    for cls in [0, 1]:
        ok = np.where((cohort.y == cls) & ((oof > 0) == (cls == 1)) & (n_crops >= 10))[0]
        ok = [i for i in ok if i not in selected]
        for i in rng.choice(ok, size=min(per_class, len(ok)), replace=False):
            selected[int(i)] = "cohort"
    print(f"slides: {len(selected)}", flush=True)
    return selected


class Ablation:
    """Oriented logit of one bag under marker / crop occlusion of its raw crops."""

    def __init__(
        self, chain: ReferenceChain, head, mu: torch.Tensor, sd: torch.Tensor, crops: torch.Tensor
    ):
        self.chain, self.head, self.mu, self.sd, self.crops = chain, head, mu, sd, crops
        self.n = crops.shape[0]
        self.z_ref = self.z(crops)  # [n, D]
        self.logit_ref = head(self.z_ref[None])[0][0, 0].item()
        self.sign = 1.0 if self.logit_ref > 0 else -1.0
        self.ref = self.logit(self.z_ref[None]).item()

    def z(self, crops: torch.Tensor) -> torch.Tensor:
        """Z-scored z_global of preprocessed crops, in batches."""
        z = torch.cat([
            self.chain.encode_crops(crops[s : s + BATCH])[0] for s in range(0, len(crops), BATCH)
        ])
        return (z - self.mu) / self.sd

    def logit(self, bags: torch.Tensor) -> torch.Tensor:
        return self.head(bags)[0][:, 0] * self.sign

    def one_crop_replaced(self, z_new: torch.Tensor) -> np.ndarray:
        """Effect of replacing crop c (alone) by z_new[c], for every c."""
        bags = self.z_ref[None].repeat(self.n, 1, 1)
        bags[torch.arange(self.n), torch.arange(self.n)] = z_new
        return (self.ref - self.logit(bags)).cpu().numpy()

    def zero_channels(self, channels: list[int]) -> tuple[float, np.ndarray]:
        """(effect of zeroing the channels in all crops, effect per single crop)."""
        ablated = self.crops.clone()
        ablated[:, channels] = 0
        z_ablated = self.z(ablated)
        return self.ref - self.logit(z_ablated[None]).item(), self.one_crop_replaced(z_ablated)

    def occlude_crops(self) -> np.ndarray:
        """Effect of zeroing each whole crop (all markers)."""
        z_zero = self.z(torch.zeros_like(self.crops[:1]))
        return self.one_crop_replaced(z_zero[0].expand(self.n, -1))


@torch.no_grad()
def ablate_slide(
    i: int, role: str, cohort, models, chain: ReferenceChain, groups: dict, out_dir: Path
) -> None:
    f = cohort.fold_of[i]
    mu, sd = (torch.from_numpy(s).float().to(DEVICE) for s in models.stats[f])
    image = cohort.names[i]
    coords = cohort.coords(i)
    density = cohort.density(i)
    slide = imc.preprocess(tifffile.imread(imc.tiff_path(image))).astype(np.float32)
    crops = torch.from_numpy(np.stack([slide[:, a : a + P, b : b + P] for a, b in coords])).to(
        DEVICE
    )  # [n, 43, P, P]
    logit_stored = models.heads[f](models.zscored_bag(cohort, i, DEVICE))[0][0, 0].item()

    abl = Ablation(chain, models.heads[f], mu, sd, crops)
    crop_occ = abl.occlude_crops()
    n_markers = crops.shape[1]
    slide_abl, crop_abl = np.zeros(n_markers), np.zeros((abl.n, n_markers))
    for m in range(n_markers):
        slide_abl[m], crop_abl[:, m] = abl.zero_channels([m])
    group_names = list(groups)
    group_abl, group_crop_abl = np.zeros(len(groups)), np.zeros((abl.n, len(groups)))
    for gi, channels in enumerate(groups.values()):
        group_abl[gi], group_crop_abl[:, gi] = abl.zero_channels(channels)

    top = np.argsort(-slide_abl)[:5]
    print(
        f"[{role}] {image} {cohort.classes[cohort.y[i]][:12]:12s} fold {f} n={abl.n} logit stored "
        f"{logit_stored:+.2f} re-enc {abl.logit_ref:+.2f} | top markers "
        + ", ".join(f"{chain.markers[m]} {slide_abl[m]:+.2f}" for m in top)
        + " | groups "
        + ", ".join(
            f"{g.split('/')[0]} {v:+.2f}" for g, v in zip(group_names, group_abl, strict=True)
        ),
        flush=True,
    )
    extra = {"slide": slide.astype(np.float16)} if role == "example" else {}
    np.savez_compressed(
        out_dir / f"{image[:-5]}.npz",
        img=image,
        label=cohort.classes[cohort.y[i]],
        y=cohort.y[i],
        role=role,
        fold=f,
        logit_stored=logit_stored,
        logit_ref=abl.logit_ref,
        coords=coords,
        density=density["density"].values,
        empty=density["empty"].values.astype(bool),
        crop_occ=crop_occ,
        slide_abl=slide_abl,
        crop_abl=crop_abl,
        group_names=np.array(group_names),
        group_abl=group_abl,
        group_crop_abl=group_crop_abl,
        marker_mean=crops.mean((2, 3)).cpu().numpy(),
        markers=np.array(chain.markers),
        **extra,
    )


def main(per_class: int, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cohort = abmil.load_cohort("DX.name")
    models = abmil.load_fold_models(cohort, DEVICE)
    oof = abmil.oof_logits(cohort, models, DEVICE)
    chain = ReferenceChain(DEVICE, with_immuvis=True)
    groups = {g: [chain.markers.index(m) for m in ms] for g, ms in style.MARKER_GROUPS.items()}
    if sorted(i for channels in groups.values() for i in channels) != list(
        range(len(chain.markers))
    ):
        raise ValueError("MARKER_GROUPS must partition the cords panel")
    for i, role in select_slides(cohort, oof, per_class).items():
        ablate_slide(i, role, cohort, models, chain, groups, out)
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--per-class", type=int, default=15, help="correctly classified cohort slides per class"
    )
    ap.add_argument("--out-dir", default=str(paths.CACHE / "marker_ablation"))
    a = ap.parse_args()
    main(a.per_class, a.out_dir)
