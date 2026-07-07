# scripts/run_abmil.py
import os
from types import SimpleNamespace

import hydra
import numpy as np
import pandas as pd
import rootutils
from omegaconf import DictConfig

# Repo convention (see src/train.py, src/eval.py, scripts/diagnose_model.py,
# scripts/encode_mil_embeddings.py): put the project root on sys.path so
# `from src...` resolves when this file is invoked directly as
# `python scripts/run_abmil.py` (its own dir would otherwise be sys.path[0]).
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.downstream.abmil.cv import run_cv
from src.downstream.abmil.data import build_image_bags

# Mirrors `/home/tnocon/mil/gated_abmil/src/run_abmil.py::drop_nan_labels` /
# `balance_meta_df`: image-level filtering on the pooled (train+test) meta
# table, applied per dataset x feature before building bags.


def drop_nan_labels(meta_df: pd.DataFrame) -> pd.DataFrame:
    valid_imgs = meta_df.dropna(subset=["feature_value"])["img_path"].unique()
    return meta_df[meta_df["img_path"].isin(valid_imgs)]


def balance_meta_df(meta_df: pd.DataFrame, min_class_freq: float = 0.05) -> pd.DataFrame:
    """Image-level class-frequency filter: keep classes with freq >= min_class_freq."""
    img_labels = meta_df.groupby("img_path", sort=False)["feature_value"].first()
    img_labels = img_labels[~pd.isna(img_labels)].astype(str)
    if len(img_labels) == 0:
        return meta_df.iloc[0:0]
    unique_classes, counts = np.unique(img_labels.values, return_counts=True)
    freq = counts / counts.sum()
    keep_classes = set(unique_classes[freq >= float(min_class_freq)])
    keep_imgs = img_labels[img_labels.isin(keep_classes)].index
    return meta_df[meta_df["img_path"].isin(keep_imgs)]


def _load_pooled_meta(meta_dir: str, dataset: str) -> pd.DataFrame:
    """Pool the train+test metadata CSVs for one dataset (CV runs over both)."""
    frames = []
    for split in ("train", "test"):
        path = os.path.join(meta_dir, f"{dataset}_{split}_metadata.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"no {dataset}_{{train,test}}_metadata.csv under {meta_dir}")
    pooled = pd.concat(frames, ignore_index=True)
    # The encode stage writes only [img_path, coords*, embeddings_file, embedding_idx];
    # `feature_value` (the clinical label) must be joined in as a separate manual step.
    # Fail loud here rather than KeyError-ing deep in drop_nan_labels.
    if "feature_value" not in pooled.columns:
        raise KeyError(
            f"{dataset} metadata under {meta_dir} has no 'feature_value' column -- join "
            f"clinical labels onto the encode output before running run_abmil (columns: "
            f"{list(pooled.columns)})."
        )
    return pooled


@hydra.main(version_base=None, config_path="../configs", config_name="downstream/abmil")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.results_dir, exist_ok=True)

    cv_fold_rows, cv_summary_rows, results_rows = [], [], []

    def _flush():
        # Incremental write: re-emit after every feature so a later crash never
        # discards already-completed features' metrics.
        pd.DataFrame(cv_fold_rows).to_csv(os.path.join(cfg.results_dir, "cv_folds.csv"), index=False)
        pd.DataFrame(cv_summary_rows).to_csv(os.path.join(cfg.results_dir, "cv_summary.csv"), index=False)
        pd.DataFrame(results_rows).to_csv(os.path.join(cfg.results_dir, "results.csv"), index=False)

    for dataset in cfg.datasets:
        pooled = _load_pooled_meta(cfg.meta_dir, dataset)
        has_feature_col = "feature" in pooled.columns
        available = set(pooled["feature"].unique()) if has_feature_col else set()
        features = [f for f in cfg.features if (f in available or not has_feature_col)]
        # Wide format (single `feature_value` column, no `feature` selector): every
        # requested feature would reuse the SAME column and produce identical metrics.
        # Only meaningful for a single feature -- reject the silent-duplicate case.
        if not has_feature_col and len(features) > 1:
            raise ValueError(
                f"{dataset} metadata has no 'feature' column but {len(features)} features "
                f"were requested ({features}); each would reuse the same 'feature_value' "
                f"column and yield identical metrics. Use a long-format CSV with a "
                f"'feature' column, or request exactly one feature."
            )

        for feature in features:
            try:
                feat_meta = pooled[pooled["feature"] == feature] if has_feature_col else pooled
                feat_meta = drop_nan_labels(feat_meta)
                feat_meta = balance_meta_df(feat_meta, min_class_freq=cfg.min_class_freq)
                if len(feat_meta) == 0:
                    print(f"[{dataset}/{feature}] skipping (no data after filtering)", flush=True)
                    continue

                classes = sorted(feat_meta["feature_value"].astype(str).unique())
                class_to_idx = {c: i for i, c in enumerate(classes)}
                num_classes = len(classes)

                bags, labels = build_image_bags(feat_meta, class_to_idx)
                if len(bags) == 0:
                    print(f"[{dataset}/{feature}] skipping (no bags)", flush=True)
                    continue

                print(f"=== {dataset}/{feature}: {len(bags)} images, {num_classes} classes ===", flush=True)

                fold_results_dir = os.path.join(cfg.results_dir, dataset, feature)
                os.makedirs(fold_results_dir, exist_ok=True)
                fold_cfg = SimpleNamespace(
                    num_folds=cfg.num_folds,
                    hidden_dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    num_epochs=cfg.num_epochs,
                    patience=cfg.patience,
                    lr=cfg.lr,
                    batch_size=cfg.batch_size,
                    zscore=cfg.zscore,
                    results_dir=fold_results_dir,
                    seed=getattr(cfg, "seed", 42),
                )

                out = run_cv(bags, labels, num_classes=num_classes, cfg=fold_cfg)
            except Exception as e:  # noqa: BLE001 -- isolate one feature's failure
                # A rare-subtype feature can trip run_cv's `min_class_count < 2`
                # ValueError (or any other per-feature error). Log and continue so a
                # single feature never aborts the whole datasets x features sweep and
                # discards already-completed metrics (which `_flush` has persisted).
                print(f"[{dataset}/{feature}] FAILED: {type(e).__name__}: {e}", flush=True)
                continue

            cv_fold_rows.extend(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "fold": fold,
                    "cv_accuracy": out["fold_accuracy"][fold],
                    "cv_macro_f1": out["fold_macro_f1"][fold],
                    "cv_auc": out["fold_auc"][fold],
                }
                for fold in range(len(out["fold_accuracy"]))
            )

            cv_summary_rows.append({
                "dataset": dataset,
                "feature": feature,
                "folds_used": out["num_folds_used"],
                "cv_accuracy_mean": float(np.mean(out["fold_accuracy"])),
                "cv_macro_f1_mean": float(np.mean(out["fold_macro_f1"])),
                "cv_auc_mean": float(np.nanmean(out["fold_auc"])),
                "oof_accuracy": out["oof_accuracy"],
                "oof_macro_f1": out["oof_macro_f1"],
                "oof_auc": out["oof_auc"],
                "n_total": len(bags),
            })

            results_rows.append({
                "dataset": dataset,
                "feature": feature,
                "accuracy": out["oof_accuracy"],
                "macro_f1": out["oof_macro_f1"],
                "auc": out["oof_auc"],
            })

            _flush()

    _flush()


if __name__ == "__main__":
    main()
