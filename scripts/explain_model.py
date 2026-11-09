"""Hydra entrypoint for post-hoc explainability on a trained PIGVAE checkpoint.

Mirrors the training/inference conventions (Hydra compose + `hydra.utils.instantiate` +
`load_state_dict(strict=False)`). Runs an attribution method on a batch of crops and writes
per-node heatmaps (+ marker attribution for the encoder IG path) to `out_dir`.

Examples:
  uv run scripts/explain_model.py experiment=vae16_fb0p0_film \
      ckpt_path=/raid_.../last.ckpt method=encoder_ig target=0 paths=szary
  uv run scripts/explain_model.py experiment=vae16_fb0p0_film \
      ckpt_path=/raid_.../last.ckpt method=rollout paths=szary
"""

import operator
import os

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("multiply", lambda x, y: operator.mul(int(x), int(y)), replace=True)
OmegaConf.register_new_resolver("divide", lambda x, y: int(x) // int(y), replace=True)

from src.explainability import viz
from src.explainability.encoder_attribution import encoder_ig, pca_to_markers
from src.explainability.rollout import attention_rollout


@hydra.main(version_base="1.3", config_path="../configs", config_name="explain")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    # ── model ──
    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[explain] loaded {cfg.ckpt_path} (epoch {ckpt.get('epoch', '?')}) "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    gae = model.graph_ae

    # ── data ──
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("fit" if cfg.split == "val" else "test")
    loader = dm.val_dataloader() if cfg.split == "val" else dm.test_dataloader()
    batch = next(iter(loader))
    nf = batch.node_features[: cfg.num_samples].float()
    print(f"[explain] method={cfg.method} on {nf.shape[0]} crops "
          f"(nodes={nf.shape[1]}, F={nf.shape[2]})")

    if cfg.method == "encoder_ig":
        attr = encoder_ig(gae, nf, target=cfg.target, n_steps=cfg.n_steps)
        node_imp = attr.abs().sum(-1)  # [B, N]
        torch.save({"attr": attr, "node_importance": node_imp}, os.path.join(cfg.out_dir, "encoder_ig.pt"))
        for b in range(nf.shape[0]):
            viz.plot_node_heatmap(node_imp[b], cfg.grid_size,
                                  os.path.join(cfg.out_dir, f"node_ig_{b}.png"),
                                  title=f"encoder IG z_global[{cfg.target}] · crop {b}")
        pca_layer = getattr(dm, "pca_layer", None)
        if cfg.to_markers and pca_layer is not None:
            marker_attr = pca_to_markers(attr, pca_layer.components)  # [B, N, C]
            torch.save(marker_attr, os.path.join(cfg.out_dir, "encoder_ig_markers.pt"))
            viz.plot_marker_attribution(marker_attr, os.path.join(cfg.out_dir, "markers_top.png"))
            print(f"[explain] marker attribution -> {marker_attr.shape[-1]} channels")

            # Reconstruct the embedding feature map (inverse-PCA) and render the 3-option panel:
            # [attribution | aggregate | false-colour] + top channel maps, per crop.
            feat = pca_layer.inverse(nf)  # [B, N, C]
            top_channels = marker_attr.abs().sum(dim=(0, 1)).argsort(descending=True).tolist()
            for b in range(nf.shape[0]):
                viz.plot_crop_panel(
                    node_imp[b], feat[b], cfg.grid_size, top_channels,
                    os.path.join(cfg.out_dir, f"panel_{b}.png"),
                    title=f"crop {b} · IG(z_global[{cfg.target}]) vs embedding projections",
                )

    elif cfg.method == "rollout":
        node_imp = attention_rollout(gae, nf)
        torch.save({"node_importance": node_imp}, os.path.join(cfg.out_dir, "rollout.pt"))
        for b in range(nf.shape[0]):
            viz.plot_node_heatmap(node_imp[b], cfg.grid_size,
                                  os.path.join(cfg.out_dir, f"node_rollout_{b}.png"),
                                  title=f"attention rollout · crop {b}")

    elif cfg.method == "faithfulness":
        # Validate the encoder-IG node map: deletion/insertion on z_global[target], IG vs random.
        from src.explainability.encoder_attribution import _encode_zglobal
        from src.explainability.faithfulness import faithfulness_score

        attr = encoder_ig(gae, nf, target=cfg.target, n_steps=cfg.n_steps)
        node_imp = attr.abs().sum(-1)  # [B, N]
        rows = []
        for b in range(nf.shape[0]):
            x = nf[b]
            fwd = lambda bag: _encode_zglobal(gae, bag.unsqueeze(0))[0, cfg.target]
            fs_ig = faithfulness_score(fwd, x, node_imp[b], steps=20)
            torch.manual_seed(b)
            fs_rand = faithfulness_score(fwd, x, torch.rand(x.shape[0]), steps=20)
            print(f"[faithfulness] crop {b}: IG={fs_ig['faithfulness']:+.4f} "
                  f"(ins {fs_ig['insertion_auc']:+.4f} - del {fs_ig['deletion_auc']:+.4f}) | "
                  f"random={fs_rand['faithfulness']:+.4f}  "
                  f"{'PASS' if fs_ig['faithfulness'] > fs_rand['faithfulness'] else 'FAIL'}")
            rows.append({"crop": b, "ig": fs_ig, "random": fs_rand})
        torch.save(rows, os.path.join(cfg.out_dir, "faithfulness.pt"))

    elif cfg.method in ("abmil_ig", "abmil_attention"):
        raise NotImplementedError(
            "ABMIL decision-level explainability needs a trained ABMIL head + per-image bag "
            "assembly. The attribution functions are ready and tested in "
            "src/explainability/abmil_attribution.py; wiring the entrypoint (load abmil_ckpt, "
            "build z_global bags per image) is the next step."
        )
    else:
        raise ValueError(f"unknown method {cfg.method!r}")

    print(f"[explain] wrote outputs to {os.path.abspath(cfg.out_dir)}")


if __name__ == "__main__":
    main()
