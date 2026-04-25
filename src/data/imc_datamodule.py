from pathlib import Path
from typing import Any

import joblib
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from tqdm import tqdm

from src.data.components.graphs_datamodules import (
    DenseGraphDataLoader,
    DualOutputTransform,
    GridGraphDataset,
    IMCBaseDictTransform,
    PatchAugmentations,
    PCADenseGraphCollator,
    PCALayer,
    PickleDataset,
    SingleViewTransform,
    WelfordOnline,
)


def resolve_imc_h5(data_dir: str | Path, imc_root: str, dataset_name: str, split: str) -> Path:
    """Resolve per-dataset HDF5 path: ``{data_dir}/{imc_root}/{dataset_name}/{split}.h5``."""
    path = Path(data_dir) / imc_root / dataset_name / f"{split}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"No HDF5 at {path}")
    return path


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

        self.single_view = hparams.get("single_view", False)

        if self.single_view:
            svt_kwargs = dict(
                center_crop_size=hparams.center_crop_size,
                normalize=hparams.normalize,
            )
            self.dual_transforms_train = DualOutputTransform(
                base_transforms=None,
                augmentation_transforms=SingleViewTransform(is_validation=False, **svt_kwargs),
            )
            self.dual_transforms_val = DualOutputTransform(
                base_transforms=None,
                augmentation_transforms=SingleViewTransform(is_validation=True, **svt_kwargs),
            )
        else:
            self.base_transforms = IMCBaseDictTransform(
                center_crop_size=hparams.center_crop_size, normalize=hparams.normalize
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
        self.num_pca_components = hparams.num_pca_components
        self.num_node_features = hparams.num_node_features
        self.center_crop_size = hparams.center_crop_size
        self.normalize = hparams.normalize
        self.clip_range = hparams.clip_range
        self.zscore = hparams.zscore

        self.imc_root = hparams.imc_root
        _extra = hparams.imc_dataset_names
        self.imc_dataset_names: list[str] = (
            list(_extra) if _extra is not None and len(_extra) > 0 else []
        )

    def _active_names(self) -> list[str]:
        return self.imc_dataset_names if self.imc_dataset_names else [self.imc_dataset_name]

    def _resolve(self, split: str) -> list[Path]:
        return [
            resolve_imc_h5(self.data_dir, self.imc_root, name, split) for name in self._active_names()
        ]

    def _load_datasets(self, paths: list[Path], transform: DualOutputTransform) -> Dataset:
        parts = [
            PickleDataset(
                path,
                transform=transform,
                generate_views=not self.single_view,
                center_crop_size=self.center_crop_size,
                single_view=self.single_view,
            )
            for path in paths
        ]
        return parts[0] if len(parts) == 1 else ConcatDataset(parts)

    def _pca_path(self) -> Path:
        return (
            Path(self.data_dir)
            / self.imc_root / self.imc_dataset_names[0]
            / f"pca_model_{self.num_pca_components}_center_crop_{self.center_crop_size}.pkl"
        )
    def _statistics_path(self) -> Path:
        return (
            Path(self.data_dir)
            / self.imc_root / self.imc_dataset_names[0]
            / f"imc_statistics_{self.num_pca_components}_center_crop_{self.center_crop_size}.pt"
        )

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
        train_paths = self._resolve("train")
        test_paths = self._resolve("test")
        for path in train_paths + test_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Expected HDF5 at {path}")

        pca_model_path = self._pca_path()
        statistics_path = self._statistics_path()

        # Compute statistics only once (on rank 0)
        if self.trainer and self.trainer.is_global_zero:
            if not pca_model_path.exists():
                dataset = self._load_datasets(
                    train_paths,
                    transform=self.dual_transforms_train,
                )

                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False,
                )

                ipca = IncrementalPCA(
                    n_components=self.num_pca_components, batch_size=self.batch_size
                )

                for batch in tqdm(loader, desc="Fitting IncrementalPCA", leave=True):
                    # Default collate stacks DualOutputTransform output: (augmented, argsort, perm, ...)
                    augmented = batch[0][: self.batch_size]
                    x = augmented.reshape(-1, self.num_channels).detach().cpu().numpy()
                    if x.shape[0] < self.num_pca_components:
                        continue
                    ipca.partial_fit(x)

                joblib.dump(ipca, str(pca_model_path))
                welford_online = WelfordOnline(self.num_pca_components)
                for batch in tqdm(loader, desc="Computing Welford Online", leave=True):
                    augmented = batch[0][: self.batch_size]
                    x = augmented.reshape(-1, self.num_channels).detach().cpu().numpy()
                    x_proj_np = ipca.transform(x)
                    x_proj_torch = torch.from_numpy(x_proj_np)
                    welford_online.update(x_proj_torch)
                mean, std = welford_online.finalize()
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
            train_paths = self._resolve("train")
            test_paths = self._resolve("test")
            trainset = self._load_datasets(train_paths, self.dual_transforms_train)
            testset = self._load_datasets(test_paths, self.dual_transforms_val)
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

        pca_model_path = self._pca_path()
        statistics_path = self._statistics_path()
        self.pca_layer = PCALayer(
            pca_model_path, statistics_path, clip_range=self.clip_range, zscore=self.zscore
        )

    def _get_collate_fn(self) -> Any:
        return PCADenseGraphCollator(self.pca_layer)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.data_train is None:
            raise RuntimeError(
                "Expected self.data_train to be set in setup() before calling train_dataloader().",
            )
        train_dataset = GridGraphDataset(
            grid_size=self.grid_size, dataset=self.data_train, channels=list(range(4))
        )

        return DenseGraphDataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            shuffle=True,
            collate_fn=self._get_collate_fn(),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        # TODO: "Using train dataloader for validation."
        # if self.data_train is None:
        #     raise RuntimeError(
        #         "Expected self.data_train to be set in setup() before calling train_dataloader().",
        #     )
        # train_dataset = GridGraphDataset(
        #     grid_size=self.grid_size, dataset=self.data_train, channels=list(range(4))
        # )

        # return DenseGraphDataLoader(
        #     dataset=train_dataset,
        #     batch_size=self.batch_size_per_device,
        #     num_workers=self.num_workers,
        #     pin_memory=self.pin_memory,
        #     persistent_workers=self.num_workers > 0,
        #     collate_fn=self._get_collate_fn(),
        #     shuffle=False,
        # )

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
            collate_fn=self._get_collate_fn(),
            shuffle=False,
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
            grid_size=self.grid_size, dataset=self.data_test, channels=list(range(1))
        )

        return DenseGraphDataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._get_collate_fn(),
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
