"""Export the self-contained model bundle served by ``src/deploy`` (run on szary).

Writes ImmuVis, the PCA-128, the PIGVAE encoder and one fold's gated-ABMIL head (with that fold's
z-score statistics) into one directory, then verifies it against the thesis pipeline:

1. ABMIL: the head reproduces the CV out-of-fold logits on every validation image of its fold
   (from the stored z_global bags). Also selects which checkpoint of the fold the CV used.
2. Whole chain: for a few validation images, ``IMCPipeline`` on the raw TIFF reproduces the
   stored per-crop z_global and the out-of-fold logit.

The export fails if a check fails. The fold defaults to the best validation AUROC.

Usage (on szary, from the repo root)::

    uv run python scripts/deploy/export_bundle.py --out /raid_.../deploy/bundle
    uv run python scripts/deploy/export_bundle.py --out ... --fold 3 --n-check 5

Then upload it (needs Modal credentials): ``modal volume put pigvae-imc-bundle <out> /``.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import rootutils
import tifffile
import torch
import yaml
from omegaconf import OmegaConf

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.deploy.pipeline import IMCPipeline
from src.downstream.abmil.cv import zscore_stats_from_bags
from src.downstream.abmil.data import (
    balance_meta_df,
    build_image_bags_with_sources,
    drop_nan_labels,
    load_pooled_meta,
)
from src.downstream.abmil.lit import AbmilLitModule
from src.downstream.encode import compose_model_config
from src.models import immuvis
from src.models.components.modules import GraphAE

# ---- sources (reference model: experiment=pigvae, W&B run 81zasevr) ---------------------------
EMB = Path(OmegaConf.load(ROOT / "configs" / "paths" / "szary.yaml").root_dir)
RAID = EMB.parents[1]
IMMUVIS_CKPT = RAID / "models" / "ImmuVis-475-768-ma1-refactored.pth"
IMMUVIS_CONFIG = RAID / "models" / "config.ImmuVis-475-768-ma1-refactored.yaml"
PCA_PKL = EMB / "data" / "IMC" / "cords" / "pca_model_128_center_crop_16.pkl"
PIGVAE_EXPERIMENT = "pigvae"
PIGVAE_CKPT = EMB / "logs" / "train" / "runs" / "2026-07-31_10-29-22" / "checkpoints" / "last.ckpt"
DOWNSTREAM = EMB / "downstream" / "vae16_s0p01_film_2026-07-31"
META_DIR = DOWNSTREAM / "meta_tables" / "zglobal"
RESULTS = DOWNSTREAM / "results" / "zglobal"
TIFF_DIRS = [RAID / "immuvis_split" / s / "cords" / "imgs" for s in ("train", "test")]
SUBMODULE_CONFIGS = ROOT / "external" / "multiplex-image-model" / "configs"

DATASET = "cords"
MIN_CLASS_FREQ = 0.05  # configs/downstream/abmil.yaml
MAX_FOLDS = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cohort(label: str):
    """Bags, labels, per-bag crop coords and CV folds, rebuilt exactly as ``run_cv`` did."""
    pooled = load_pooled_meta(str(META_DIR), DATASET)
    meta = balance_meta_df(
        drop_nan_labels(pooled[pooled["feature"] == label]), min_class_freq=MIN_CLASS_FREQ
    )
    classes = sorted(meta["feature_value"].astype(str).unique())
    bags, labels, coords, _ = build_image_bags_with_sources(
        meta, {c: i for i, c in enumerate(classes)}
    )
    names = [g["img_path"].iloc[0] for _, g in meta.groupby("img_path", sort=False)]
    y = labels.numpy()
    n_folds = min(MAX_FOLDS, int(np.bincount(y).min()), len(bags))
    folds = list(
        StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(
            np.zeros(len(bags)), y
        )
    )
    return classes, bags, y, coords, names, folds


def head_logits(model: torch.nn.Module, bags, idx, mu, sd) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in idx:
            x = torch.from_numpy((np.asarray(bags[i], np.float32) - mu) / sd)[None]
            out.append(model(x)[0][0].numpy())
    return np.stack(out)


def export_abmil(label: str, fold: int | None, out: Path) -> dict:
    classes, bags, y, coords, names, folds = cohort(label)
    oof = np.load(RESULTS / DATASET / label / "oof_predictions.npz", allow_pickle=True)
    oof_logits = oof["logits"]
    if list(oof["classes"]) != classes or len(oof_logits) != len(bags):
        raise SystemExit(
            f"OOF file does not match the rebuilt cohort: classes {list(oof['classes'])} vs "
            f"{classes}, {len(oof_logits)} vs {len(bags)} images"
        )

    def fold_auc(val):
        s = oof_logits[val]
        if s.shape[1] == 1:
            return float(roc_auc_score(y[val], s[:, 0]))
        p = torch.softmax(torch.from_numpy(s), 1).numpy()
        return float(roc_auc_score(y[val], p, multi_class="ovr"))

    aucs = [fold_auc(val) for _, val in folds]
    print(f"[abmil] {label}: {len(bags)} images, {len(folds)} folds, classes {classes}")
    print("[abmil] fold validation AUROC: " + " ".join(f"{a:.3f}" for a in aucs))
    fold = int(np.argmax(aucs)) if fold is None else fold
    train, val = folds[fold]
    mu, sd = zscore_stats_from_bags(bags, indices=train)

    # The fold directory holds the checkpoint of each CV run; keep the one the reported OOF used.
    best = None
    for ckpt in sorted((RESULTS / DATASET / label / f"fold_{fold}").glob("*.ckpt")):
        lit = AbmilLitModule.load_from_checkpoint(str(ckpt), map_location="cpu").eval()
        err = float(np.abs(head_logits(lit.model, bags, val, mu, sd) - oof_logits[val]).max())
        print(f"[abmil] fold {fold} {ckpt.name}: max |logit - OOF| = {err:.2e}")
        if best is None or err < best[0]:
            best = (err, ckpt, lit)
    err, ckpt, lit = best
    if err > 1e-3:
        raise SystemExit(f"no checkpoint of fold {fold} reproduces the OOF logits (best {err:.2e})")

    hp = lit.hparams
    fname = f"abmil_{label}.pt"
    torch.save(
        {
            "hparams": {
                "emb_dim": hp.emb_dim,
                "hidden_dim": hp.hidden_dim,
                "num_heads": hp.num_heads,
                "num_classes": hp.num_classes,
            },
            "state_dict": lit.model.state_dict(),
            "mu": torch.from_numpy(mu),
            "sd": torch.from_numpy(sd),
            "classes": classes,
            "label": label,
            "fold": fold,
            "fold_val_auroc": aucs[fold],
        },
        out / fname,
    )
    return {
        "file": fname,
        "fold": fold,
        "ckpt": ckpt,
        "fold_aurocs": aucs,
        "check_images": [(names[i], bags[i], coords[i], oof_logits[i]) for i in val],
    }


def export_pigvae(out: Path) -> None:
    ctx = compose_model_config(PIGVAE_EXPERIMENT)
    hparams = OmegaConf.to_container(ctx.model.graph_ae.hparams, resolve=True)
    gae = GraphAE(OmegaConf.create(hparams))
    state = torch.load(PIGVAE_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    sub = {k.removeprefix("graph_ae."): v for k, v in state.items() if k.startswith("graph_ae.")}
    gae.load_state_dict(sub, strict=True)
    torch.save({"hparams": hparams, "state_dict": gae.state_dict()}, out / "pigvae.pt")


def export_immuvis(out: Path) -> None:
    ckpt = torch.load(IMMUVIS_CKPT, map_location="cpu", weights_only=False)
    torch.save({"model_state_dict": ckpt["model_state_dict"]}, out / "immuvis.pth")  # no optimizer
    shutil.copyfile(IMMUVIS_CONFIG, out / "immuvis.yaml")
    _, (missing, unexpected) = immuvis.load_immuvis(out / "immuvis.pth", out / "immuvis.yaml")
    if missing or unexpected:
        raise SystemExit(f"ImmuVis: missing {missing[:3]}, unexpected {unexpected[:3]}")
    ids, names = immuvis.channel_ids(
        SUBMODULE_CONFIGS / "all_panels_config.yaml",
        SUBMODULE_CONFIGS / "all_markers_tokenizer.yaml",
        DATASET,
    )
    (out / "markers.json").write_text(
        json.dumps({"panel": DATASET, "markers": names, "channel_ids": ids.tolist()}, indent=1)
    )


def export_pca(out: Path) -> None:
    pca = joblib.load(PCA_PKL)
    torch.save(
        {
            "components": torch.tensor(pca.components_, dtype=torch.float32),
            "mean": torch.tensor(pca.mean_, dtype=torch.float32),
        },
        out / "pca.pt",
    )


def find_tiff(name: str) -> Path:
    for d in TIFF_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def check_chain(out: Path, images: list, n: int, device: str) -> None:
    """IMCPipeline on raw TIFFs vs the stored z_global and the OOF logit."""
    pipe = IMCPipeline.from_bundle(out, device=device)
    for name, bag, coords, oof_logit in images[:n]:
        res = pipe.predict(tifffile.imread(find_tiff(name)), ig_steps=0, include_embeddings=True)
        by_pos = {(c["row"], c["col"]): c["z_global"] for c in res["crops"]}
        stored_pos = [tuple(int(v) for v in rc) for rc in coords]
        if set(by_pos) != set(stored_pos):
            raise SystemExit(f"{name}: crop grid differs ({len(by_pos)} vs {len(stored_pos)})")
        z = np.array([by_pos[p] for p in stored_pos], np.float32)
        stored = np.asarray(bag, np.float32)
        corr = float(np.corrcoef(z.ravel(), stored.ravel())[0, 1])
        z_err = float(np.abs(z - stored).max())
        logit = np.atleast_1d(np.asarray(res["logit"], np.float32))
        l_err = float(np.abs(logit - oof_logit).max())
        print(
            f"[chain] {name}: {len(z)} crops | z_global corr {corr:.6f} max|d| {z_err:.2e} | "
            f"logit {logit.round(4).tolist()} vs OOF {oof_logit.round(4).tolist()} (|d| {l_err:.2e})"
        )
        if corr < 0.9999 or l_err > 1e-2:
            raise SystemExit(f"{name}: the bundle does not reproduce the thesis pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="bundle directory to write")
    parser.add_argument("--label", default="DX.name")
    parser.add_argument("--fold", type=int, default=None, help="default: best validation AUROC")
    parser.add_argument("--n-check", type=int, default=3, help="raw images to run end to end")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    export_immuvis(out)
    export_pca(out)
    export_pigvae(out)
    ab = export_abmil(args.label, args.fold, out)

    files = sorted(p.name for p in out.iterdir() if p.name != "manifest.json")
    hashes = {f: sha256(out / f) for f in files}
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "bundle_id": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "abmil": ab["file"],
        "label": args.label,
        "fold": ab["fold"],
        "fold_val_aurocs": [round(a, 4) for a in ab["fold_aurocs"]],
        "sources": {
            "immuvis": str(IMMUVIS_CKPT),
            "pca": str(PCA_PKL),
            "pigvae": f"{PIGVAE_CKPT} (experiment={PIGVAE_EXPERIMENT})",
            "abmil": str(ab["ckpt"]),
        },
        "sha256": hashes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(yaml.safe_dump({k: manifest[k] for k in ("bundle_id", "label", "fold", "abmil")}))

    check_chain(out, ab["check_images"], args.n_check, args.device)
    size = sum(p.stat().st_size for p in out.iterdir()) / 2**20
    print(f"[export] OK: {out} ({size:.0f} MiB), bundle {manifest['bundle_id']}")


if __name__ == "__main__":
    main()
