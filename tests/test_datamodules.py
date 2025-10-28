from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize
from omegaconf import open_dict


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the
    necessary attributes were created (e.g., the dataloader objects), and that dtypes
    and batch sizes correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    # Instantiate datamodule via Hydra config
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml")
        with open_dict(cfg):
            cfg.data.hparams.data_dir = data_dir
            cfg.data.hparams.batch_size = batch_size
            cfg.data.hparams.num_workers = 0
            cfg.data.hparams.pin_memory = False
        dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    # DenseGraphBatch checks
    assert hasattr(batch, "node_features")
    assert hasattr(batch, "mask")
