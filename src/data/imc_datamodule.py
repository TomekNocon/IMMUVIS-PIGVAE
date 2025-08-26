from typing import Any, Dict, Optional

import torch
from pathlib import Path
from omegaconf import DictConfig
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from src.data.components.graphs_datamodules import (
    IMCBaseDictTransform,
    PatchAugmentations,
    GridGraphDataset,
    DenseGraphDataLoader,
    DualOutputTransform,
    PickleDataset
)


class IMCDataModule(LightningDataModule):
    """`LightningDataModule` for the IMC embeddings dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(self, hparams: DictConfig) -> None:
        """Initialize a `IMCDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.base_transforms = IMCBaseDictTransform()

        self.aug_transforms_train = PatchAugmentations(
            prob=hparams.augmentation_prob,
            size=hparams.size,
            patch_size=hparams.patch_size,
        )

        self.aug_transforms_val = PatchAugmentations(
            prob=hparams.augmentation_prob,
            size=hparams.size,
            patch_size=hparams.patch_size,
            is_validation=True,
        )

        self.dual_transforms_train = DualOutputTransform(
            self.base_transforms, self.aug_transforms_train
        )

        self.dual_transforms_val = DualOutputTransform(
            self.base_transforms, self.aug_transforms_val
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size = hparams.batch_size
        self.batch_size_per_device = self.batch_size
        self.data_dir = hparams.data_dir
        self.train_val_test_split = hparams.train_val_test_split
        self.grid_size = hparams.grid_size
        self.num_workers = hparams.num_workers
        self.pin_memory = hparams.pin_memory
        self.is_contrastive = hparams.is_contrastive
        self.num_aug_per_sample = hparams.num_aug_per_sample

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of IMC classes (not specified).
        """
        return -1

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        path = Path(self.data_dir) / "IMC-sample" / "test_1.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Expected dataset at {path}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        # TODO: do the transforms only for train and validation
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_path = Path(self.data_dir) / "IMC-sample" / "train.pkl"
            test_path = Path(self.data_dir) / "IMC-sample" / "test.pkl"
            trainset = PickleDataset(
                train_path, transform=self.dual_transforms_train
            )
            testset = PickleDataset(
                test_path, transform=self.dual_transforms_val
            )
            train_ratio, val_ratio, test_ratio, leftover_ratio = (
                self.train_val_test_split
            )
            size_testset = len(testset)
            size_trainset = len(trainset)
            self.data_train, _ = random_split(
                dataset=trainset,
                lengths=[train_ratio, size_trainset - train_ratio],
                generator=torch.Generator().manual_seed(42),
            )
            # dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_val, self.data_test, _ = random_split(
                dataset=testset,
                lengths=[val_ratio, test_ratio, size_testset - val_ratio - test_ratio],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        assert self.data_train is not None
        train_dataset = GridGraphDataset(
            grid_size=self.grid_size, dataset=self.data_train, channels=[0]
        )

        return DenseGraphDataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        assert self.data_val is not None
        val_dataset = GridGraphDataset(
            grid_size=self.grid_size, dataset=self.data_val, channels=[0]
        )

        return DenseGraphDataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        assert self.data_test is not None
        test_dataset = GridGraphDataset(
            grid_size=self.grid_size, dataset=self.data_test, channels=[0]
        )

        return DenseGraphDataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


def add_channel(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0)


if __name__ == "__main__":
    _ = MNISTDataModule()
