"""Extract arrays for the MNIST / Fashion-MNIST reconstruction figure (Sec. 5.2.1).

Re-runs the PIGVAE validation forward pass from a trained checkpoint, exactly as
`PLGraphAE.validation_step` / `on_validation_epoch_end` do when they log the wandb
"Prediction" / "Ground Truth" / "Perms" panels (training=True, tau from the scheduler at
the checkpoint epoch), and caches:

  views      [n, 8, 24, 24]  input D4 views as fed to the encoder (not re-oriented)
  pred       [n, 8, 24, 24]  reconstruction, restored to the canonical orientation (as logged)
  gt         [n, 8, 24, 24]  ground truth, restored the same way (as logged)
  perm       [n, 8, 36, 36]  permutation matrix chosen by the permuter
  perm_cls   [n, 8]          index of the chosen predefined D4 matrix
  perm_prob  [n, 8, 8]       permuter soft probabilities
  correct    [n, 8]          perm_cls == majority class for that view slot (whole val set)

plus whole-validation-set D4 view accuracy (soft and hard forward).

Must run with the MNIST-era code (worktree of the `mnist` / `fashion-mnist` branch), on a node that
mounts /raid_encrypted, e.g.:
  cd /home/tnocon/master_thesis/pigvae-mnist && .venv/bin/python \
     /home/tnocon/master_thesis/IMMUVIS-PIGVAE/scripts/thesis_figures/extract_mnist_recon.py \
     --code-root . --run-dir <hydra run dir> --out <npz>
"""
import argparse, json, os, sys

import numpy as np
import torch


def main(a):
    code_root = os.path.abspath(a.code_root)
    os.environ["PROJECT_ROOT"] = code_root
    sys.path.insert(0, code_root)
    os.chdir(code_root)
    import hydra
    from omegaconf import OmegaConf
    import src.models.components.metrics.recontructions as R
    import src.models.components.plot as pL
    from src.models.pigvae_auto_module import PLGraphAE

    torch.manual_seed(a.seed)
    cfg = OmegaConf.load(os.path.join(a.run_dir, ".hydra", "config.yaml"))
    cfg.paths.root_dir = code_root
    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    dm.setup("fit")

    ckpt = os.path.join(a.run_dir, "checkpoints", a.ckpt)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    epoch = int(ck["epoch"])
    model = PLGraphAE.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval()
    tau = float(model.temperature_scheduler(epoch))
    P = 8

    soft_cls, hard_cls, first = [], [], None
    with torch.no_grad():
        for b, graph in enumerate(dm.val_dataloader()):
            graph_emb, graph_pred, soft_probs, perm, mu, logvar = model(graph=graph, training=True, tau=tau)
            _, _, soft_probs_h, _, _, _ = model(graph=graph, training=False, tau=1.0)
            bs = graph.node_features.shape[0] // P
            # batch layout: index = view_slot * bs + image
            soft_cls.append(soft_probs.argmax(-1).view(P, bs).T)
            hard_cls.append(soft_probs_h.argmax(-1).view(P, bs).T)
            if b == 0:
                first = (graph, graph_pred, soft_probs, perm, bs)

    soft_cls = torch.cat(soft_cls).numpy()  # [N_val, 8]
    hard_cls = torch.cat(hard_cls).numpy()
    acc = {}
    for name, cls in [("soft", soft_cls), ("hard", hard_cls)]:
        maj = np.array([np.bincount(cls[:, k], minlength=P).argmax() for k in range(P)])
        acc[name] = {
            "slot_to_class": maj.tolist(),
            "bijective": bool(len(set(maj.tolist())) == P),
            "accuracy": float((cls == maj[None]).mean()),
            "per_slot": (cls == maj[None]).mean(0).round(4).tolist(),
            "all8_correct_frac": float((cls == maj[None]).all(1).mean()),
            "n_images": int(cls.shape[0]),
        }
    maj_soft = np.array(acc["soft"]["slot_to_class"])

    # Frame-independent ("equivariance") accuracy: the model may pick a different canonical
    # orientation per image, i.e. view k gets class  A[k] * g  for an image-specific D4 element g
    # (A = majority mapping). Per image, choose the g (and composition side) that explains most
    # of its 8 views; a view is correct iff it fits that offset.
    Pdef = model.graph_ae.permuter.predefined_permutations.float()
    flat = Pdef.reshape(P, -1)
    def idx_of(m):
        hit = torch.nonzero((flat == m.reshape(1, -1)).all(1)).flatten()
        return int(hit[0]) if len(hit) else -1
    mult = np.array([[idx_of(Pdef[x] @ Pdef[y]) for y in range(P)] for x in range(P)])  # mult[x,y] = x*y
    if (mult < 0).any():
        raise RuntimeError("predefined permutations are not closed under composition")

    def equiv_correct(cls, A):
        cands = np.stack([np.stack([mult[A, g] for g in range(P)]),        # A[k]*g
                          np.stack([mult[g, A] for g in range(P)])])       # g*A[k]
        cands = cands.reshape(2 * P, P)                                     # [16, 8]
        match = cls[:, None, :] == cands[None]                              # [N, 16, 8]
        best = match.sum(-1).argmax(1)
        return match[np.arange(len(cls)), best], best

    for name, cls in [("soft", soft_cls), ("hard", hard_cls)]:
        ok, best = equiv_correct(cls, np.array(acc[name]["slot_to_class"]))
        acc[name]["equiv_accuracy"] = float(ok.mean())
        acc[name]["equiv_all8_correct_frac"] = float(ok.all(1).mean())
        acc[name]["equiv_offset_counts"] = np.bincount(best, minlength=2 * P).tolist()

    graph, graph_pred, soft_probs, perm, bs = first
    n = a.n_examples
    idx = R.batch_augmented_indices(bs, num_permutations=P, n_examples=n)  # image-major, 8 views each
    argsort = graph.argsort_augmented_features[idx]
    side, patch = cfg.data.hparams.size, cfg.data.hparams.patch_size

    def to_img(nf, restore):
        if restore:
            nf = torch.stack([x[s, :] for x, s in zip(nf, argsort)], 0)
        return pL.restore_tensor(nf, nf.shape[0], 1, side, side, patch).squeeze(1).numpy().reshape(n, P, side, side)

    out = dict(
        views=to_img(graph.node_features[idx], False),
        pred=to_img(graph_pred.node_features[idx], True),
        gt=to_img(graph.node_features[idx], True),
        perm=perm[idx].numpy().reshape(n, P, *perm.shape[1:]),
        perm_cls=soft_probs[idx].argmax(-1).numpy().reshape(n, P),
        perm_prob=soft_probs[idx].numpy().reshape(n, P, P),
        labels=graph.y[:n].numpy() if getattr(graph, "y", None) is not None else np.full(n, -1),
    )
    out["correct_majority"] = out["perm_cls"] == maj_soft[None]
    out["correct"], _ = equiv_correct(out["perm_cls"], maj_soft)
    meta = dict(run_dir=a.run_dir, ckpt=a.ckpt, epoch=epoch, tau=tau, d4_accuracy=acc,
                view_order=["r0", "r0_hflip", "r90", "r90_hflip", "r180", "r180_hflip", "r270", "r270_hflip"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, meta=json.dumps(meta), **out)
    print(json.dumps(meta, indent=1))
    print("saved", a.out, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="last.ckpt")
    ap.add_argument("--n-examples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
