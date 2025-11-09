from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig

from src.data.components.graphs_datamodules import DenseGraphBatch
from src.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the
    necessary attributes were created (e.g., the dataloader objects), and that dtypes
    and batch sizes correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    hparams = DictConfig({
        "size": 28,
        "patch_size": 4,
        "augmentation_prob": 0.0,
        "grid_size": 7,
        "num_workers": 0,
        "pin_memory": False,
        "is_contrastive": False,
        "num_aug_per_sample": 8,
        "data_dir": data_dir,
        "batch_size": batch_size,
        "train_val_test_split": (55_000, 5_000, 10_000, 0),
    })
    dm = MNISTDataModule(hparams=hparams)
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
    assert isinstance(batch, DenseGraphBatch)
    assert isinstance(batch.node_features, torch.Tensor)
    assert batch.node_features.dtype == torch.float32
