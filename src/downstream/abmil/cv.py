"""Stratified k-fold CV runner for the downstream ABMIL classifier.

Ports `zscore_stats_from_bags` / `apply_zscore_to_bags` and the fold-loop
structure from `/home/tnocon/mil/gated_abmil/src/run_abmil.py::run`, swapping
the bespoke `GatedABMILClassifierWithValidation` train loop for Task B4's
`AbmilLitModule` + `pl.Trainer(EarlyStopping, ModelCheckpoint)`.

Leakage guardrail (Global Constraints): `zscore='cv_train'` fits mean/std on
the TRAIN fold only and applies it to the val fold -- normalization is never
fit on pooled train+val.
"""

import os

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.downstream.abmil.data import MILDataset, mil_collate
from src.downstream.abmil.lit import AbmilLitModule


def zscore_stats_from_bags(bags, indices=None, eps=1e-6):
    """Per-dimension mean/std across all crop rows in the selected bags."""
    chunks = bags if indices is None else [bags[int(i)] for i in indices]
    if len(chunks) == 0:
        raise ValueError("zscore_stats_from_bags: empty bag selection")
    stacked = np.concatenate([np.asarray(b) for b in chunks], axis=0).astype(np.float64, copy=False)
    mu = stacked.mean(axis=0)
    std = np.maximum(stacked.std(axis=0), float(eps))
    return mu.astype(np.float32), std.astype(np.float32)


def apply_zscore_to_bags(bags, mu, std):
    """Return new bags with shape-preserving z-score `(x - mu) / std`."""
    mu = np.asarray(mu, dtype=np.float32).reshape(1, -1)
    std = np.asarray(std, dtype=np.float32).reshape(1, -1)
    return [((np.asarray(b, dtype=np.float32) - mu) / std).astype(np.float32) for b in bags]


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))


def _auc(y_true, logits, num_classes):
    y_true = np.asarray(y_true).reshape(-1).astype(int, copy=False)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    logits = np.asarray(logits)
    if int(num_classes) <= 2:
        scores = _sigmoid_np(logits.reshape(-1))
        try:
            return float(roc_auc_score(y_true, scores))
        except ValueError:
            return float("nan")
    if logits.ndim != 2:
        return float("nan")
    z = logits - logits.max(axis=1, keepdims=True)
    expz = np.exp(z)
    probs = expz / expz.sum(axis=1, keepdims=True)
    try:
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


@torch.no_grad()
def _predict(lit_module, dl, num_classes):
    lit_module.eval()
    device = next(lit_module.model.parameters()).device
    all_logits, all_y = [], []
    for bags, mask, y in dl:
        logits, _ = lit_module.model(bags.to(device), mask.to(device))
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())
    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_y).numpy()
    if num_classes <= 2:
        preds = (_sigmoid_np(logits.reshape(-1)) > 0.5).astype(int)
    else:
        preds = logits.argmax(axis=1)
    return y_true, preds, logits


def run_cv(bags, labels, num_classes, cfg):
    """Stratified k-fold CV over `bags`/`labels` for a single dataset x feature.

    Per fold: builds `MILDataset`/`DataLoader` (collate=`mil_collate`), an
    `AbmilLitModule`, and a `pl.Trainer` with `EarlyStopping('val_loss',
    patience)` + `ModelCheckpoint('val_loss')`; the best checkpoint is reloaded
    for val-fold inference. `cfg.zscore == 'cv_train'` fits mean/std on the
    train fold only (see module docstring).

    Returns a dict with per-fold `fold_accuracy`/`fold_macro_f1`/`fold_auc`
    lists, pooled out-of-fold (`oof_*`) metrics/arrays, and `num_folds_used`.
    """
    labels_t = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
    labels_np = labels_t.detach().cpu().numpy().astype(int)
    n_bags = len(bags)

    _, class_counts = np.unique(labels_np, return_counts=True)
    min_class_count = int(class_counts.min()) if len(class_counts) else 0
    num_folds = min(int(cfg.num_folds), min_class_count, n_bags)
    if num_folds < 2:
        raise ValueError(
            f"run_cv: cannot form >=2 folds (num_folds={cfg.num_folds}, "
            f"min_class_count={min_class_count}, n_bags={n_bags})"
        )

    zscore = getattr(cfg, "zscore", "none")
    if zscore == "global":
        # Leakage-prone: fits mean/std on ALL bags (train+val pooled) -- prefer 'cv_train'.
        g_mu, g_std = zscore_stats_from_bags(bags)
        bags_for_split = apply_zscore_to_bags(bags, g_mu, g_std)
    elif zscore in ("cv_train", "none"):
        bags_for_split = bags
    else:
        raise ValueError("cfg.zscore must be one of: none, global, cv_train")

    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_accuracy, fold_macro_f1, fold_auc = [], [], []
    oof_pred = np.full(n_bags, -1, dtype=int)
    logits_dim = 1 if num_classes <= 2 else int(num_classes)
    oof_logits = np.full((n_bags, logits_dim), np.nan, dtype=np.float32)

    results_dir = getattr(cfg, "results_dir", ".")
    emb_dim = bags[0].shape[1]

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(n_bags), labels_np)):
        if zscore == "cv_train":
            # Leakage guardrail: fit mean/std on the TRAIN fold only, apply to both splits.
            mu, std = zscore_stats_from_bags(bags, indices=train_idx)
            bags_norm = apply_zscore_to_bags(bags, mu, std)
        else:
            bags_norm = bags_for_split

        train_bags = [bags_norm[i] for i in train_idx]
        val_bags = [bags_norm[i] for i in val_idx]

        train_ds = MILDataset(train_bags, labels_t[train_idx])
        val_ds = MILDataset(val_bags, labels_t[val_idx])
        train_dl = DataLoader(train_ds, batch_size=int(cfg.batch_size), shuffle=True, collate_fn=mil_collate)
        val_dl = DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False, collate_fn=mil_collate)

        lit_module = AbmilLitModule(
            emb_dim=emb_dim,
            hidden_dim=int(cfg.hidden_dim),
            num_heads=int(cfg.num_heads),
            num_classes=num_classes,
            lr=float(cfg.lr),
        )

        fold_dir = os.path.join(str(results_dir), f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        ckpt_cb = ModelCheckpoint(dirpath=fold_dir, monitor="val_loss", mode="min", save_top_k=1)
        es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=int(cfg.patience))

        trainer = Trainer(
            max_epochs=int(cfg.num_epochs),
            callbacks=[es_cb, ckpt_cb],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator="auto",
            devices=1,
        )
        trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

        if ckpt_cb.best_model_path:
            lit_module = AbmilLitModule.load_from_checkpoint(ckpt_cb.best_model_path)

        y_true, preds, logits = _predict(lit_module, val_dl, num_classes)
        acc = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, average="macro"))
        auc = _auc(y_true, logits, num_classes)

        fold_accuracy.append(acc)
        fold_macro_f1.append(f1)
        fold_auc.append(auc)

        oof_pred[np.asarray(val_idx)] = preds.astype(int, copy=False)
        v_logits = logits if logits.ndim == 2 else logits.reshape(-1, 1)
        oof_logits[np.asarray(val_idx)] = v_logits.astype(np.float32, copy=False)

    eval_mask = oof_pred != -1
    oof_accuracy = float(accuracy_score(labels_np[eval_mask], oof_pred[eval_mask]))
    oof_macro_f1 = float(f1_score(labels_np[eval_mask], oof_pred[eval_mask], average="macro"))
    oof_auc = _auc(labels_np[eval_mask], oof_logits[eval_mask], num_classes)

    return {
        "fold_accuracy": fold_accuracy,
        "fold_macro_f1": fold_macro_f1,
        "fold_auc": fold_auc,
        "oof_accuracy": oof_accuracy,
        "oof_macro_f1": oof_macro_f1,
        "oof_auc": oof_auc,
        "oof_predictions": oof_pred[eval_mask],
        "oof_logits": oof_logits[eval_mask],
        "oof_ground_truth": labels_np[eval_mask],
        "num_folds_used": num_folds,
    }
