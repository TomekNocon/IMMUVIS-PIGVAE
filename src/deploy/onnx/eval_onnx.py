import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import onnxruntime as ort
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def evaluate(
    session: ort.InferenceSession,
    dm: Any,
    limit: int = 50,
) -> dict[str, float]:
    _ = {i.name: i for i in session.get_inputs()}
    outputs = [o.name for o in session.get_outputs()]
    total_mse = 0.0
    n = 0

    dm.prepare_data()
    dm.setup(stage="validate")
    loader = dm.val_dataloader()

    for i, batch in enumerate(loader):
        if i >= limit:
            break
        batch = batch.to(torch.device("cpu"))
        node_features = batch.node_features.numpy()
        mask = batch.mask.numpy()
        pred_logits = session.run(
            outputs,
            {"node_features": node_features, "mask": mask},
        )[0]
        gt = batch.node_features.numpy()
        total_mse += float(np.mean((pred_logits - gt) ** 2))
        n += 1
    return {"mse": total_mse / max(1, n)}


@hydra.main(
    version_base="1.3",
    config_path="../../../configs",
    config_name="deploy/onnx/eval.yaml",
)
def main(cfg: DictConfig) -> None:
    onnx_path = cfg.get("onnx_path")
    if not onnx_path:
        raise ValueError("Provide onnx_path=... to evaluate ONNX model")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dm: Any = hydra.utils.instantiate(cfg.data)
    metrics = evaluate(sess, dm, limit=int(cfg.get("limit", 50)))
    log.info(f"ONNX val metrics: {metrics}")
    Path(onnx_path).with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2),
    )


if __name__ == "__main__":
    main()
