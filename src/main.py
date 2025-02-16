from typing import Any, Dict, List, Optional, Tuple


import os # or pathlib
import logging

import hydra
import lightning as L
import rootutils
from lightning import ModelCheckpoint, LearningRateMonitor, Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger, WandbLogger
from torchvision import transforms, datasets
from omegaconf import DictConfig
import torch
from data.components.graphs_datamodules import SplitPatches, GraphDataModule, SIZE, PATCH_SIZE
from models.pigvae_auto_module import PLGraphAE
from models.components.losses import Critic
from pigvae.synthetic_graphs.custom_hyperparameter import add_arguments
from argparse import ArgumentParser

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

# https://github.com/Lightning-AI/pytorch-lightning/discussions/16688

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict

logging.getLogger("lightning").setLevel(logging.WARNING)

def main(hparams):
    # Create directories if they do not exist
    run_dir = os.path.join(hparams.save_dir, f"run{hparams.id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Starting Run {hparams.id}, checkpoints will be saved in {run_dir}")

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="{epoch}-{val_loss:.2f}",
        save_last=True,  # Save the last checkpoint
        save_top_k=1,    # Save the best checkpoint based on `monitor`
        monitor="val_loss",  # Metric to monitor
        mode="min"       # Save the checkpoint with the lowest validation loss
    )

    # Define learning rate monitor
    lr_logger = LearningRateMonitor(logging_interval="step")

    # Define Wandb Logger
    wandb_logger = WandbLogger(project=hparams.wb_project_name, log_model=True)
    wandb_logger.experiment.config.update(vars(hparams))

    # Define model
    critic = Critic
    model = PLGraphAE(hparams.__dict__, critic)

    # Define data transformation
    train_transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        SplitPatches(PATCH_SIZE)
    ])

    # Load MNIST dataset
    mnist = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
    ]),
    )

    train_subset_indices = list(range(100))  # Adjust sample size if needed
    val_subset_indices = list(range(100, 110)) 
    train_graph_kwargs_grid = {
        "grid_size": 6,
        "imgs": mnist.data[train_subset_indices].unsqueeze(1),
        "targets": mnist.targets[train_subset_indices],
        "img_transform": train_transform,
        "channels": [0]
    }
    
    val_graph_kwargs_grid = {
        "grid_size": 6,
        "imgs": mnist.data[val_subset_indices].unsqueeze(1),
        "targets": mnist.targets[val_subset_indices],
        "img_transform": train_transform,
        "channels": [0]
    }
    
    # Define data module
    datamodule = GraphDataModule(
        graph_family=hparams.graph_family,
        train_graph_kwargs=train_graph_kwargs_grid,
        val_graph_kwargs=val_graph_kwargs_grid,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        samples_per_epoch=100,
        distributed_sampler=None
    )
    
    
    # Define trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        val_check_interval= 1, # hparams.eval_freq if not hparams.test else 100,
        accelerator="cpu",
        callbacks=[lr_logger, checkpoint_callback],
        precision=hparams.precision,
        max_epochs=hparams.num_epochs,
        log_every_n_steps=5,
        # gradient_clip_val=0.1,
        # terminate_on_nan=True,
        # reload_dataloaders_every_epoch=True, # https://github.com/Lightning-AI/pytorch-lightning/discussions/7372
        # resume_from_checkpoint=hparams.resume_ckpt if hparams.resume_ckpt != "" else None
    )

    # Train the model
    wandb_logger.watch(model)
    trainer.fit(model=model, datamodule=datamodule)

    # Save the best checkpoint path to W&B
    wandb_logger.experiment.log({"best_model_path": checkpoint_callback.best_model_path})
    wandb_logger.experiment.unwatch(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args)
    
    

@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()


