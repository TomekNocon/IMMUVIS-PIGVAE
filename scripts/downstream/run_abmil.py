"""Gated-ABMIL transfer evaluation with 10-fold CV (stage 3 of transfer).

Builds per-image bags from the label-joined meta tables, runs stratified 10-fold CV per
dataset x clinical target, and writes fold, summary and result CSVs (plus OOF predictions)
to ``results_dir``.

Usage:
    uv run python scripts/downstream/run_abmil.py run_tag=<RUN_TAG> source_tag=zglobal
"""

import os
from types import SimpleNamespace

import hydra
import numpy as np
import pandas as pd
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.downstream.abmil.cv import run_cv
from src.downstream.abmil.data import (
    balance_meta_df,
    build_image_bags,
    drop_nan_labels,
    load_pooled_meta,
)


@hydra.main(version_base=None, config_path="../../configs", config_name="downstream/abmil")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.results_dir, exist_ok=True)

    cv_fold_rows, cv_summary_rows, results_rows = [], [], []

    def _flush():
        # Incremental write: re-emit after every feature so a later crash never
        # discards already-completed features' metrics.
        pd.DataFrame(cv_fold_rows).to_csv(
            os.path.join(cfg.results_dir, "cv_folds.csv"), index=False
        )
        pd.DataFrame(cv_summary_rows).to_csv(
            os.path.join(cfg.results_dir, "cv_summary.csv"), index=False
        )
        pd.DataFrame(results_rows).to_csv(os.path.join(cfg.results_dir, "results.csv"), index=False)

    for dataset in cfg.datasets:
        pooled = load_pooled_meta(cfg.meta_dir, dataset)
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

                print(
                    f"=== {dataset}/{feature}: {len(bags)} images, {num_classes} classes ===",
                    flush=True,
                )

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
            except Exception as e:
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
                "oof_precision": out["oof_precision"],
                "oof_recall": out["oof_recall"],
                "oof_auc": out["oof_auc"],
                "n_total": len(bags),
            })

            results_rows.append({
                "dataset": dataset,
                "feature": feature,
                "accuracy": out["oof_accuracy"],
                "macro_f1": out["oof_macro_f1"],
                "precision": out["oof_precision"],
                "recall": out["oof_recall"],
                "auc": out["oof_auc"],
            })

            # Persist the raw OOF predictions so any further metric never needs a CV re-run.
            np.savez(
                os.path.join(fold_results_dir, "oof_predictions.npz"),
                ground_truth=out["oof_ground_truth"],
                predictions=out["oof_predictions"],
                logits=out["oof_logits"],
                classes=np.array(classes, dtype=object),
            )

            _flush()

    _flush()


if __name__ == "__main__":
    main()
