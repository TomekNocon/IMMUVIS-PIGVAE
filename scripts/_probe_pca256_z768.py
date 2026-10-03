import operator

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import torch  # noqa: F401
from hydra import compose, initialize
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("multiply", lambda x, y: operator.mul(int(x), int(y)))
OmegaConf.register_new_resolver("divide", lambda x, y: int(x) // int(y))

with initialize(version_base="1.3", config_path="../configs"):
    cfg = compose(
        config_name="train",
        overrides=["experiment=vae16_fb0p0_film_pca256_z768", "trainer=gpu", "paths=szary"],
    )

# Instantiate datamodule + model. Do NOT call dm.setup() by hand — the PCA-256 model
# is fit inside prepare_data() (rank-0, guarded by `not exists`), which Lightning runs
# for us inside trainer.fit(). Poking setup() first would only try to LOAD a pkl that
# doesn't exist yet.
dm = hydra.utils.instantiate(cfg.data)
model = hydra.utils.instantiate(cfg.model)

# Widened dims are live before any data touches the model.
enc = model.graph_ae.encoder
assert enc.projection_in.in_features == 256, enc.projection_in.in_features
assert enc.projection_in.out_features == 768, enc.projection_in.out_features
print("[probe] MODEL OK (projection_in 256->768, z_global width 768)")

import wandb

wandb.init(mode="disabled")

from lightning import Trainer

# fast_dev_run drives the real pipeline: prepare_data (fits pca_model_256_center_crop_16.pkl,
# one-time) -> setup (loads it) -> one train+val step. Catches OOM at hidden 768.
trainer = Trainer(
    fast_dev_run=1, accelerator="auto", devices=1, logger=False, enable_checkpointing=False
)
trainer.fit(model, datamodule=dm)
print("[probe] FAST-DEV-RUN OK (PCA-256 fit + one train+val step, finite, no OOM at hidden 768)")
print("[probe] ALL-OK")
