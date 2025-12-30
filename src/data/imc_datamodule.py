from pathlib import Path
from typing import Any

import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.components.graphs_datamodules import (
    DenseGraphDataLoader,
    DualOutputTransform,
    GridGraphDataset,
    IMCBaseDictTransform,
    PatchAugmentations,
    PickleDataset,
    WelfordOnline,
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

        self.base_transforms = IMCBaseDictTransform(
            center_crop_size=hparams.center_crop_size, normalize=True
        )

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

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.batch_size = hparams.batch_size
        self.batch_size_per_device = self.batch_size
        self.data_dir = hparams.data_dir
        self.train_val_test_split = hparams.train_val_test_split
        self.grid_size = hparams.grid_size
        self.num_workers = hparams.num_workers
        self.pin_memory = hparams.pin_memory
        self.is_contrastive = hparams.is_contrastive
        self.num_aug_per_sample = hparams.num_aug_per_sample
        self.num_channels = hparams.num_channels

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of IMC classes (not specified).
        """
        return -1

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is
        called only within a single process on CPU, so you can safely add your
        downloading logic within. In case of multi-node training, the execution of this
        hook depends upon `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        train_path = Path(self.data_dir) / "IMC" / "nsclc2_panel1_train.h5"
        test_path = Path(self.data_dir) / "IMC" / "nsclc2_panel1_test.h5"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Expected dataset at {train_path} and {test_path}")

        statistics_path = Path(self.data_dir) / "IMC" / "imc_statistics.pt"

        # Compute statistics only once (on rank 0)
        if self.trainer and self.trainer.is_global_zero:
            if not statistics_path.exists():
                # Load raw dataset (no normalization!)
                dataset = PickleDataset(
                    train_path, transform=self.dual_transforms_train, only_embeddings=True
                )

                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False,
                )

                welford = WelfordOnline(self.num_channels)

                for node_features in loader:
                    welford.update(node_features)

                mean, std = welford.finalize()
                torch.save({"mean": mean, "std": std}, statistics_path)

        # DDP sync
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`,
        `self.data_test`.

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
            train_path = Path(self.data_dir) / "IMC" / "nsclc2_panel1_train.h5"
            test_path = Path(self.data_dir) / "IMC" / "nsclc2_panel1_test.h5"
            # statistics_path = Path(self.data_dir) / "IMC" / "imc_statistics.pt"
            # stats = torch.load(statistics_path, map_location="cpu")
            # mean = stats["mean"]
            # std = stats["std"]
            # self.dual_transforms_train.set_mean(mean)
            # self.dual_transforms_train.set_std(std)
            # self.dual_transforms_val.set_mean(mean)
            # self.dual_transforms_val.set_std(std)
            trainset = PickleDataset(train_path, transform=self.dual_transforms_train)
            testset = PickleDataset(test_path, transform=self.dual_transforms_val)
            train_ratio, val_ratio, test_ratio, _ = self.train_val_test_split
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
        if self.data_train is None:
            raise RuntimeError(
                "Expected self.data_train to be set in setup() before calling train_dataloader().",
            )
        train_dataset = GridGraphDataset(
            grid_size=self.grid_size, dataset=self.data_train, channels=list(range(64))
        )

        return DenseGraphDataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if self.data_val is None:
            raise RuntimeError(
                "Expected self.data_val to be set in setup() before calling val_dataloader().",
            )
        val_dataset = GridGraphDataset(
            grid_size=self.grid_size, dataset=self.data_val, channels=list(range(64))
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
        if self.data_test is None:
            raise RuntimeError(
                "Expected self.data_test to be set in setup() before calling test_dataloader().",
            )
        test_dataset = GridGraphDataset(
            grid_size=self.grid_size, dataset=self.data_test, channels=list(range(64))
        )

        return DenseGraphDataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the
        datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given
        datamodule `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """


def add_channel(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0)


if __name__ == "__main__":
    pass
