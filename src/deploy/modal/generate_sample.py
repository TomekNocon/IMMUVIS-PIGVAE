import json
from pathlib import Path

import hydra
import rootutils
import torch
from hydra import compose, initialize

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def main() -> None:
    # Load the same config tree used by eval/export, then instantiate via DictConfig
    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(config_name="deploy/onnx/eval.yaml")

    # Force batch_size=1 for a simple payload conforming to the exported model
    cfg.data.hparams.batch_size = 1

    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    dm.setup(stage="validate")
    batch = next(iter(dm.val_dataloader()))
    batch = batch.to(torch.device("cpu"))

    node_features = batch.node_features.numpy()
    if getattr(batch, "mask", None) is not None:
        mask = batch.mask.numpy()
    else:
        # If mask wasn't provided, use all-true mask
        b, n, _ = node_features.shape
        mask = torch.ones((b, n), dtype=torch.bool).numpy()

    payload = {
        "node_features": node_features.tolist(),
        "mask": mask.tolist(),
    }

    out_path = Path("src/deploy/modal/payload.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"Wrote {out_path} with shapes: node_features={node_features.shape}, mask={mask.shape}")


if __name__ == "__main__":
    main()
