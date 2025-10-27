from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import operator

import rootutils


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

OmegaConf.register_new_resolver("multiply", lambda x, y: operator.mul(int(x), int(y)))
OmegaConf.register_new_resolver("divide", lambda x, y: int(x) // int(y))

from src.data.components.graphs_datamodules import DenseGraphBatch
from src.utils import RankedLogger, extras, task_wrapper


log = RankedLogger(__name__, rank_zero_only=True)


class GraphAEExportWrapper(nn.Module):
    """
    Tensor-only forward for ONNX export.
    Inputs:
      - node_features: (B, N, Din)
      - mask: (B, N)
    Outputs:
      - node_logits: (B, N, Dout)
      - graph_emb: (B, Demb)
    """

    def __init__(self, graph_ae: nn.Module) -> None:
        super().__init__()
        self.graph_ae = graph_ae

    def forward(
        self, node_features: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = node_features.device
        edge_features = torch.empty(0, device=device)
        batch = DenseGraphBatch(
            node_features=node_features,
            edge_features=edge_features,
            mask=mask,
        )
        graph_emb, graph_pred, _, _, _, _ = self.graph_ae(
            graph=batch, training=False, tau=1.0
        )
        return graph_pred.node_features, graph_emb


@task_wrapper
def export(cfg: DictConfig) -> None:
    if not cfg.get("ckpt_path") and not cfg.get("ckpt_artifact"):
        raise ValueError("Provide ckpt_path=... or ckpt_artifact=entity/project:name")

    onnx_path: Optional[str] = cfg.get("onnx_path")
    if onnx_path is None:
        out_dir = Path(cfg.paths.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = str(out_dir / "model.onnx")

    datamodule = hydra.utils.instantiate(cfg.data)
    pl_module = hydra.utils.instantiate(cfg.model)

    ckpt_path: Optional[str] = cfg.get("ckpt_path")
    if not ckpt_path and cfg.get("ckpt_artifact"):
        
        run = wandb.init(
            project=cfg.get("wandb_project", None),
            entity=cfg.get("wandb_entity", None),
            job_type="export",
            settings=wandb.Settings(start_method="thread"),
        )
        art = run.use_artifact(cfg.ckpt_artifact, type="model")
        ckpt_dir = art.download()
        candidates = ["last.ckpt", "best.ckpt"]
        local = None
        for c in candidates:
            p = Path(ckpt_dir) / c
            if p.exists():
                local = str(p)
                break
        if local is None:
            ckpts = list(Path(ckpt_dir).rglob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError("No .ckpt found in downloaded artifact")
            local = str(ckpts[0])
        ckpt_path = local

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = pl_module.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning(f"Missing keys when loading state_dict: {len(missing)}")
    if unexpected:
        log.warning(f"Unexpected keys when loading state_dict: {len(unexpected)}")

    pl_module.eval()
    graph_ae = pl_module.graph_ae
    graph_ae.eval()

    if hasattr(graph_ae.permuter, "break_symmetry_scale"):
        graph_ae.permuter.break_symmetry_scale = 0.0

    datamodule.prepare_data()
    datamodule.setup(stage="validate")
    val_loader = datamodule.val_dataloader()
    example_batch: DenseGraphBatch = next(iter(val_loader))
    example_batch = example_batch.to(torch.device("cpu"))
    node_features = example_batch.node_features
    mask = example_batch.mask

    wrapper = GraphAEExportWrapper(graph_ae).cpu()

    dynamic_axes = {
        "node_features": {0: "batch"},
        "mask": {0: "batch"},
        "node_logits": {0: "batch"},
        "graph_emb": {0: "batch"},
    }

    input_names = ["node_features", "mask"]
    output_names = ["node_logits", "graph_emb"]

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (node_features, mask),
            onnx_path,
            export_params=True,
            opset_version=17, # important for modules like GELU or LayerNorm
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    log.info(f"ONNX model saved to: {onnx_path}")

    if bool(cfg.get("wandb_log", False)):
        import json
        import subprocess
        import wandb

        metadata = {
            "opset_version": 17,
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
            "node_features_shape": tuple(node_features.shape),
            "mask_shape": tuple(mask.shape),
            "torch_version": torch.__version__,
        }
        try:
            git_sha = (
                __import__("subprocess").check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            )
            metadata["git_commit"] = git_sha
        except Exception:
            pass

        run = wandb.init(
            project=cfg.get("wandb_project", None),
            entity=cfg.get("wandb_entity", None),
            job_type="export",
            settings=wandb.Settings(start_method="thread"),
        )
        art_name = cfg.get("onnx_artifact_name", None) or "model-onnx"
        artifact = wandb.Artifact(art_name, type="model", metadata=metadata)
        artifact.add_file(onnx_path)
        if cfg.get("ckpt_artifact"):
            try:
                artifact.add_reference(f"wandb-artifact://{cfg.ckpt_artifact}")
            except Exception:
                pass
        run.log_artifact(artifact)
        log.info(f"Logged ONNX artifact to W&B: {run.project}/{art_name}")

    # satisfy @task_wrapper contract
    return {}, {"onnx_path": onnx_path}


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="deploy/onnx/export.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    export(cfg)

if __name__ == "__main__":
    main()


