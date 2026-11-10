"""MNIST datamodule built from the Hydra data config."""

from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize
from omegaconf import open_dict


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Downloads MNIST, splits it as configured and yields D4-augmented node-grid batches."""
    data_dir = "data/"
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml")
    with open_dict(cfg):
        cfg.data.hparams.data_dir = data_dir
        cfg.data.hparams.batch_size = batch_size
        cfg.data.hparams.num_workers = 0
        cfg.data.hparams.pin_memory = False
    hp = cfg.data.hparams
    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()
    train, val, test, _ = hp.train_val_test_split
    assert (len(dm.data_train), len(dm.data_val), len(dm.data_test)) == (train, val, test)

    batch = next(iter(dm.train_dataloader()))
    num_nodes = (hp.size // hp.patch_size) ** 2
    assert batch.node_features.shape[:2] == (batch_size * hp.num_aug_per_sample, num_nodes)
    assert batch.mask.shape == batch.node_features.shape[:2]
