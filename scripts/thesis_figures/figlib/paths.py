"""Every location the figure scripts read from or write to, defined once.

Raid locations derive from ``configs/paths/szary.yaml`` so they follow the training config.
"""

from pathlib import Path

import rootutils
from omegaconf import OmegaConf

REPO = rootutils.find_root(__file__, indicator=".project-root")
FIGURES = REPO / "figures"
CACHE = FIGURES / "_cache"

# ---- raid -------------------------------------------------------------------------------------
EMBEDDINGS = Path(OmegaConf.load(REPO / "configs" / "paths" / "szary.yaml").root_dir)
RAID = EMBEDDINGS.parents[1]  # /raid_encrypted/immucan
CORDS_DATA = EMBEDDINGS / "data" / "IMC" / "cords"
PCA_128 = CORDS_DATA / "pca_model_128_center_crop_16.pkl"

TIFF_FULL = RAID / "IMC" / "immu-vis-other-ds" / "full-20122025"
TIFF_SPLIT = RAID / "immuvis_split"
CORDS_TRAIN_TIFFS = TIFF_SPLIT / "train" / "cords" / "imgs"

# ---- reference model: experiment=pigvae, wandb run 81zasevr -----------------------------------
REF_EXPERIMENT = "pigvae"
REF_WANDB_ID = "81zasevr"
REF_RUN_DIR = EMBEDDINGS / "logs" / "train" / "runs" / "2026-07-31_10-29-22"
REF_CKPT = REF_RUN_DIR / "checkpoints" / "last.ckpt"
REF_DOWNSTREAM = EMBEDDINGS / "downstream" / "vae16_s0p01_film_2026-07-31"
REF_META = REF_DOWNSTREAM / "meta_tables" / "zglobal"
REF_RESULTS = REF_DOWNSTREAM / "results" / "zglobal"

# ---- ImmuVis encoder --------------------------------------------------------------------------
IMMUVIS_SUBMODULE = REPO / "external" / "multiplex-image-model"
PANEL_CONFIG = IMMUVIS_SUBMODULE / "configs" / "all_panels_config.yaml"
MARKER_TOKENIZER = IMMUVIS_SUBMODULE / "configs" / "all_markers_tokenizer.yaml"
# The submodule is pinned to the refactored ImmuVis code (branch abmil), which together with this
# checkpoint reproduces the stored 16x16x768 embeddings (corr 1.0000 on 4 crops of 7064).
IMMUVIS_CKPT = RAID / "models" / "ImmuVis-475-768-ma1-refactored.pth"
IMMUVIS_CONFIG = RAID / "models" / "config.ImmuVis-475-768-ma1-refactored.yaml"


def figure(name: str) -> Path:
    """Output stem ``figures/<name>`` (no extension; ``io.save`` writes .pdf and .png)."""
    return FIGURES / name
