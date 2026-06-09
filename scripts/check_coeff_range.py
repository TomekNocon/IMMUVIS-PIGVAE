"""Report the magnitude range of the PCA coefficients fed to the model.

Answers: was clip_range=10 ever biting? Builds the datamodule exactly like
diagnose_model.py (data=mnist, paths=szary), iterates a few val batches, and
reports max|coeff|, per-component max, and the fraction of coefficients beyond
±3 and ±10. Run on a node that can see the data mount:

    uv run scripts/check_coeff_range.py --batches 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import rootutils
import torch

project_root = Path(__file__).resolve().parents[1]
rootutils.setup_root(str(project_root), indicator=".project-root", pythonpath=True)

from hydra.utils import instantiate
from omegaconf import OmegaConf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="szary")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batches", type=int, default=50, help="how many batches to scan")
    args = ap.parse_args()

    configs = project_root / "configs"
    data_cfg = OmegaConf.load(configs / "data" / "mnist.yaml")
    paths_cfg = OmegaConf.load(configs / "paths" / f"{args.paths}.yaml")
    for name, fn in [("multiply", lambda x, y: int(x) * int(y)),
                     ("divide", lambda x, y: int(x) // int(y))]:
        OmegaConf.register_new_resolver(name, fn, replace=True)

    # data_dir is `${paths.root_dir}/data/`; root_dir is a literal. Build it directly to
    # avoid resolving the rest of paths (output_dir uses the hydra: resolver, unavailable here).
    root_dir = OmegaConf.select(paths_cfg, "root_dir")
    data_dir = f"{root_dir}/data/"
    OmegaConf.update(data_cfg, "hparams.data_dir", data_dir)
    print(f"[config] data_dir={data_dir}")
    clip = OmegaConf.select(data_cfg, "hparams.clip_range")
    zscore = OmegaConf.select(data_cfg, "hparams.zscore")
    print(f"[config] clip_range={clip}  zscore={zscore}")
    if clip:
        print("[warn] clip_range is set — coefficients will be CLAMPED; set it to null "
              "to measure the true unclipped range.")

    dm = instantiate(data_cfg)
    dm.setup(stage="fit" if args.split == "val" else "test")
    dl = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()

    global_max = 0.0
    per_comp_max: torch.Tensor | None = None
    n_total = 0
    n_gt3 = 0
    n_gt10 = 0
    for i, batch in enumerate(dl):
        if i >= args.batches:
            break
        z = batch.node_features.detach().float()          # [B, N, k] coefficients
        a = z.abs()
        global_max = max(global_max, a.max().item())
        cmax = a.reshape(-1, a.shape[-1]).max(dim=0).values  # [k]
        per_comp_max = cmax if per_comp_max is None else torch.maximum(per_comp_max, cmax)
        n_total += a.numel()
        n_gt3 += (a > 3).sum().item()
        n_gt10 += (a > 10).sum().item()

    print(f"\nscanned {min(args.batches, i + 1)} batches, {n_total:,} coefficients")
    print(f"max|coeff|            = {global_max:.4f}")
    print(f"frac |coeff| > 3      = {100 * n_gt3 / n_total:.4f}%")
    print(f"frac |coeff| > 10     = {100 * n_gt10 / n_total:.6f}%")
    if per_comp_max is not None:
        top = torch.topk(per_comp_max, k=min(10, per_comp_max.numel()))
        print("\nper-component max|coeff| (top 10 by index:value):")
        for idx, val in zip(top.indices.tolist(), top.values.tolist(), strict=True):
            print(f"  comp {idx:3d}: {val:.4f}")
        print(f"\ncomponents whose max exceeds 10: "
              f"{(per_comp_max > 10).sum().item()} / {per_comp_max.numel()}")
        print(f"components whose max exceeds  3: "
              f"{(per_comp_max > 3).sum().item()} / {per_comp_max.numel()}")


if __name__ == "__main__":
    main()
