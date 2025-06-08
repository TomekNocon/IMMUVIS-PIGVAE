from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from src.data.components.graphs_datamodules import (
    SplitPatches,
    ImageAugmentations,
    GridGraphDataset,
    DenseGraphDataLoader,
    DualOutputTransform,
)


class MNISTDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

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
        """Initialize a `MNISTDataModule`.

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

        self.base_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Resize((hparams.size, hparams.size)),
                add_channel,
            ]
        )

        self.aug_transforms_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Resize((hparams.size, hparams.size)),
                add_channel,
                ImageAugmentations(prob=hparams.augmentation_prob),
            ]
        )

        self.aug_transforms_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Resize((hparams.size, hparams.size)),
                add_channel,
                ImageAugmentations(prob=hparams.augmentation_prob, is_validation=True),
            ]
        )

        self.patch_transform = SplitPatches(hparams.patch_size)

        self.dual_transforms_train = DualOutputTransform(
            self.base_transforms, self.aug_transforms_train, self.patch_transform
        )

        self.dual_transforms_val = DualOutputTransform(
            self.base_transforms, self.aug_transforms_val, self.patch_transform
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

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

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
            trainset = MNIST(
                self.data_dir, train=True, transform=self.dual_transforms_train
            )
            testset = MNIST(
                self.data_dir, train=False, transform=self.dual_transforms_val
            )
            train_ratio, val_ratio, test_ratio = self.train_val_test_split
            size_testset = len(testset)
            size_trainset = len(trainset)
            self.data_train, self.data_test = random_split(
                dataset=trainset,
                lengths=[train_ratio, size_trainset - train_ratio],
                generator=torch.Generator().manual_seed(42),
            )
            # dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_val, _ = random_split(
                dataset=testset,
                lengths=[val_ratio, size_testset - val_ratio],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        train_dataset = GridGraphDataset(
            grid_size=self.grid_size, dataset=self.data_train, channels=[0]
        )

        return DenseGraphDataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size_per_device
            if not self.is_contrastive
            else self.batch_size_per_device // self.num_aug_per_sample,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        val_dataset = GridGraphDataset(
            grid_size=self.grid_size, dataset=self.data_val, channels=[0]
        )

        return DenseGraphDataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size_per_device
            if not self.is_contrastive
            else self.batch_size_per_device // self.num_aug_per_sample,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        test_dataset = GridGraphDataset(
            grid_size=self.grid_size, dataset=self.data_test, channels=[0]
        )

        return DenseGraphDataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size_per_device
            if not self.is_contrastive
            else self.batch_size_per_device // self.num_aug_per_sample,
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
