"""Extract the arrays for the MNIST / Fashion-MNIST reconstruction figure (Sec. 5.2.1).

Re-runs the PIGVAE validation forward pass from a trained checkpoint exactly as
``PLGraphAE.validation_step`` / ``on_validation_epoch_end`` do for the wandb "Prediction" /
"Ground Truth" / "Perms" panels (training=True, tau from the scheduler at the checkpoint epoch):

    views      [n, 8, 24, 24]  input D4 views as fed to the encoder (not re-oriented)
    pred       [n, 8, 24, 24]  reconstruction, restored to the canonical orientation (as logged)
    gt         [n, 8, 24, 24]  ground truth, restored the same way
    perm       [n, 8, 36, 36]  permutation matrix chosen by the permuter
    perm_cls   [n, 8]          index of the chosen predefined D4 matrix
    perm_prob  [n, 8, 8]       permuter soft probabilities
    correct    [n, 8]          view consistent with the image's other views (frame-independent)
    correct_majority [n, 8]    perm_cls == the majority class of its view slot

plus whole-validation-set D4 view accuracy (soft and hard forward) in ``meta``.

Runs the model code of the ``mnist`` / ``fashion-mnist`` branch (``--code-root``, a worktree), so it
deliberately imports nothing from this repo. Launch with scripts/slurm/mnist_extract.sh.

Usage:
    cd ../pigvae-mnist && .venv/bin/python <this repo>/scripts/thesis_figures/extract_mnist_recon.py \
        --code-root . --run-dir <hydra run dir> --out <npz>
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

N_VIEWS = 8
VIEW_ORDER = ["r0", "r0_hflip", "r90", "r90_hflip", "r180", "r180_hflip", "r270", "r270_hflip"]


def load(code_root: str, run_dir: str, ckpt_name: str):
    """Datamodule, model and temperature of a run, built with the worktree's own code and config."""
    os.environ["PROJECT_ROOT"] = code_root
    sys.path.insert(0, code_root)
    os.chdir(code_root)
    import hydra
    from omegaconf import OmegaConf

    from src.models.pigvae_auto_module import PLGraphAE

    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    cfg.paths.root_dir = code_root
    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    dm.setup("fit")
    ckpt = os.path.join(run_dir, "checkpoints", ckpt_name)
    epoch = int(torch.load(ckpt, map_location="cpu", weights_only=False)["epoch"])
    model = PLGraphAE.load_from_checkpoint(ckpt, map_location="cpu").eval()
    return cfg, dm, model, epoch, float(model.temperature_scheduler(epoch))


@torch.no_grad()
def predicted_classes(model, dm, tau: float):
    """Chosen D4 class per (image, view slot) over the validation set, soft and hard forward,
    plus the outputs of the first batch."""
    soft, hard, first = [], [], None
    for b, graph in enumerate(dm.val_dataloader()):
        _, graph_pred, soft_probs, perm, _, _ = model(graph=graph, training=True, tau=tau)
        soft_probs_hard = model(graph=graph, training=False, tau=1.0)[2]
        bs = graph.node_features.shape[0] // N_VIEWS  # batch layout: index = view_slot * bs + image
        soft.append(soft_probs.argmax(-1).view(N_VIEWS, bs).T)
        hard.append(soft_probs_hard.argmax(-1).view(N_VIEWS, bs).T)
        if b == 0:
            first = (graph, graph_pred, soft_probs, perm, bs)
    return torch.cat(soft).numpy(), torch.cat(hard).numpy(), first


def composition_table(model) -> np.ndarray:
    """mult[x, y] = index of the predefined matrix x @ y (the D4 group table)."""
    predefined = model.graph_ae.permuter.predefined_permutations.float()
    flat = predefined.reshape(N_VIEWS, -1)

    def index_of(m):
        hit = torch.nonzero((flat == m.reshape(1, -1)).all(1)).flatten()
        return int(hit[0]) if len(hit) else -1

    mult = np.array([
        [index_of(predefined[x] @ predefined[y]) for y in range(N_VIEWS)] for x in range(N_VIEWS)
    ])
    if (mult < 0).any():
        raise RuntimeError("predefined permutations are not closed under composition")
    return mult


def equivariance_correct(cls: np.ndarray, majority: np.ndarray, mult: np.ndarray):
    """A view is correct if it fits ``majority[k] * g`` (or ``g * majority[k]``) for the image's best
    D4 offset g: the model may pick a different canonical orientation per image."""
    candidates = np.stack([
        np.stack([mult[majority, g] for g in range(N_VIEWS)]),
        np.stack([mult[g, majority] for g in range(N_VIEWS)]),
    ]).reshape(2 * N_VIEWS, N_VIEWS)
    match = cls[:, None, :] == candidates[None]  # [N, 16, 8]
    best = match.sum(-1).argmax(1)
    return match[np.arange(len(cls)), best], best


def d4_accuracy(soft: np.ndarray, hard: np.ndarray, mult: np.ndarray) -> dict:
    accuracy = {}
    for name, cls in [("soft", soft), ("hard", hard)]:
        majority = np.array([
            np.bincount(cls[:, k], minlength=N_VIEWS).argmax() for k in range(N_VIEWS)
        ])
        ok = cls == majority[None]
        equiv, best = equivariance_correct(cls, majority, mult)
        accuracy[name] = {
            "slot_to_class": majority.tolist(),
            "bijective": bool(len(set(majority.tolist())) == N_VIEWS),
            "accuracy": float(ok.mean()),
            "per_slot": ok.mean(0).round(4).tolist(),
            "all8_correct_frac": float(ok.all(1).mean()),
            "n_images": int(cls.shape[0]),
            "equiv_accuracy": float(equiv.mean()),
            "equiv_all8_correct_frac": float(equiv.all(1).mean()),
            "equiv_offset_counts": np.bincount(best, minlength=2 * N_VIEWS).tolist(),
        }
    return accuracy


def examples(cfg, first, n: int, majority: np.ndarray, mult: np.ndarray) -> dict:
    """Images, permutations and reconstructions of the first ``n`` validation images."""
    import src.models.components.metrics.recontructions as recon
    import src.models.components.plot as plot

    graph, graph_pred, soft_probs, perm, bs = first
    idx = recon.batch_augmented_indices(bs, num_permutations=N_VIEWS, n_examples=n)  # image-major
    argsort = graph.argsort_augmented_features[idx]
    side, patch = cfg.data.hparams.size, cfg.data.hparams.patch_size

    def to_images(node_features, restore: bool):
        if restore:
            node_features = torch.stack(
                [x[s, :] for x, s in zip(node_features, argsort, strict=True)], 0
            )
        return (
            plot
            .restore_tensor(node_features, node_features.shape[0], 1, side, side, patch)
            .squeeze(1)
            .numpy()
            .reshape(n, N_VIEWS, side, side)
        )

    out = {
        "views": to_images(graph.node_features[idx], False),
        "pred": to_images(graph_pred.node_features[idx], True),
        "gt": to_images(graph.node_features[idx], True),
        "perm": perm[idx].numpy().reshape(n, N_VIEWS, *perm.shape[1:]),
        "perm_cls": soft_probs[idx].argmax(-1).numpy().reshape(n, N_VIEWS),
        "perm_prob": soft_probs[idx].numpy().reshape(n, N_VIEWS, N_VIEWS),
        "labels": graph.y[:n].numpy() if getattr(graph, "y", None) is not None else np.full(n, -1),
    }
    out["correct_majority"] = out["perm_cls"] == majority[None]
    out["correct"], _ = equivariance_correct(out["perm_cls"], majority, mult)
    return out


def main(
    code_root: str, run_dir: str, ckpt_name: str, n_examples: int, seed: int, out: str
) -> None:
    out = os.path.abspath(out)
    torch.manual_seed(seed)
    cfg, dm, model, epoch, tau = load(os.path.abspath(code_root), run_dir, ckpt_name)
    soft, hard, first = predicted_classes(model, dm, tau)
    mult = composition_table(model)
    accuracy = d4_accuracy(soft, hard, mult)
    arrays = examples(cfg, first, n_examples, np.array(accuracy["soft"]["slot_to_class"]), mult)
    meta = {
        "run_dir": run_dir,
        "ckpt": ckpt_name,
        "epoch": epoch,
        "tau": tau,
        "d4_accuracy": accuracy,
        "view_order": VIEW_ORDER,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    print(json.dumps(meta, indent=1))
    print("saved", out, {k: v.shape for k, v in arrays.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--code-root", required=True, help="worktree of the mnist / fashion-mnist branch"
    )
    ap.add_argument("--run-dir", required=True, help="hydra run dir with checkpoints/ and .hydra/")
    ap.add_argument("--ckpt", default="last.ckpt")
    ap.add_argument("--n-examples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.code_root, a.run_dir, a.ckpt, a.n_examples, a.seed, a.out)
