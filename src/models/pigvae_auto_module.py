import os
from collections.abc import Callable
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
import wandb
from torch.optim.lr_scheduler import OneCycleLR

import src.models.components.plot as pL
from src.data.components.graphs_datamodules import DenseGraphBatch
from src.models.components.warmups import get_cosine_schedule_with_warmup

rootutils.setup_root(os.getcwd(), indicator=".project-root", pythonpath=True)


# https://stackoverflow.com/questions/65807601/output-prediction-of-pytorch-lightning-model


class PLGraphAE(L.LightningModule):
    """Example of a `LightningModule`.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        graph_ae: torch.nn.Module,
        critic: torch.nn.Module,
        kld_alpha_scheduler: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        compile: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["graph_ae", "critic", "kld_alpha_scheduler"],
            logger=False,
        )
        self.graph_ae = graph_ae
        self.critic = critic
        self.kld_alpha_scheduler = kld_alpha_scheduler
        self.automatic_optimization = True
        self.validation_step_outputs: list[dict[str, Any]] = []
        self.test_step_outputs: list[dict[str, Any]] = []

    def forward(self, graph: DenseGraphBatch) -> tuple:
        graph_emb, graph_pred, mu, logvar = self.graph_ae(graph)
        return graph_emb, graph_pred, mu, logvar

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

    def model_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

    def training_step(self, graph: DenseGraphBatch, batch_idx: int) -> torch.Tensor:
        alpha = self.kld_alpha_scheduler(self.current_epoch)
        graph_emb, graph_pred, mu, logvar = self(graph=graph)
        loss = self.critic(
            graph_emb=graph_emb,
            graph_true=graph,
            graph_pred=graph_pred,
            kld_alpha=alpha,
            mu=mu,
            logvar=logvar,
        )
        self.log_dict(loss)
        if mu is not None:
            self._log_latent_stats(mu, logvar, alpha, prefix="")
        return loss["loss"]

    def _log_latent_stats(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kld_alpha: float,
        prefix: str = "",
    ) -> None:
        bs = mu.shape[0]
        std = (0.5 * logvar).exp()
        per_dim_kld = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)  # [B, D]
        per_dim_kld_mean = per_dim_kld.mean(0)  # [D] — avg over batch
        p = f"{prefix}latent/" if prefix else "latent/"
        self.log(f"{p}mu_mean", mu.mean(), batch_size=bs)
        self.log(f"{p}mu_std", mu.std(), batch_size=bs)
        self.log(f"{p}std_mean", std.mean(), batch_size=bs)
        self.log(f"{p}active_dims", (per_dim_kld_mean > 0.1).float().sum(), batch_size=bs)
        self.log(f"{p}kld_per_dim_median", per_dim_kld_mean.median(), batch_size=bs)
        self.log(f"{p}kld_alpha", kld_alpha, batch_size=bs)

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."

    def validation_step(self, graph: DenseGraphBatch, batch_idx: int) -> dict[str, Any]:
        alpha = self.kld_alpha_scheduler(self.current_epoch)
        graph_emb, graph_pred, mu, logvar = self(graph=graph)
        outputs = {
            "prediction": graph_pred,
            "ground_truth": graph,
            "graph_emb": graph_emb,
        }
        self.validation_step_outputs.append(outputs)
        batch_size = graph_pred.node_features.shape[0]
        metrics = self.critic.evaluate(
            graph_emb=graph_emb,
            graph_true=graph,
            graph_pred=graph_pred,
            kld_alpha=alpha,
            mu=mu,
            logvar=logvar,
            prefix="val",
        )
        metrics["alpha"] = alpha
        self.log_dict(
            metrics,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
        )
        if mu is not None:
            self._log_latent_stats(mu, logvar, alpha, prefix="val_")
        return metrics

    def on_validation_epoch_end(self) -> None:
        if self.trainer.is_global_zero and wandb.run is not None:
            n_examples = 4
            predictions = self.validation_step_outputs[0]["prediction"].node_features
            ground_truths = self.validation_step_outputs[0]["ground_truth"].node_features
            graph_emb = self.validation_step_outputs[0]["graph_emb"]

            batch_size = predictions.shape[0]
            n_show = min(n_examples, batch_size)

            pred_imgs = predictions[:n_show, :, :].detach().cpu()
            gt_imgs = ground_truths[:n_show, :, :].detach().cpu()
            diff = pred_imgs - gt_imgs

            pred_min, pred_max = pred_imgs.min().item(), pred_imgs.max().item()
            gt_min, gt_max = gt_imgs.min().item(), gt_imgs.max().item()
            vmin = min(pred_min, gt_min)
            vmax = max(pred_max, gt_max)
            diff_abs_max = diff.abs().max().item()

            fig_prediction = pL.plot_feature_map(pred_imgs, n_show, vmin=vmin, vmax=vmax)
            fig_ground_truth = pL.plot_feature_map(gt_imgs, n_show, vmin=vmin, vmax=vmax)
            fig_diff = pL.plot_feature_map(diff, n_show, vmin=-diff_abs_max, vmax=diff_abs_max)

            all_embs = torch.cat(
                [el["graph_emb"] for el in self.validation_step_outputs], dim=0
            ).detach().cpu().float().numpy()
            all_targets = torch.cat(
                [el["ground_truth"].y for el in self.validation_step_outputs]
            ).numpy()
            fig_pca = pL.plot_pca(all_embs, all_targets, n_rows=100, n_cols=8)

            wandb.log({
                "Predictions": [
                    wandb.Image(fig, caption=f"Predictions {i + 1}")
                    for i, fig in enumerate(fig_prediction)
                ],
                "Ground Truth": [
                    wandb.Image(fig, caption=f"Ground Truth {i + 1}")
                    for i, fig in enumerate(fig_ground_truth)
                ],
                "Diff": [
                    wandb.Image(fig, caption=f"Diff {i + 1}")
                    for i, fig in enumerate(fig_diff)
                ],
                "PCA": wandb.Image(fig_pca, caption="PCA"),
            })
            for fig in fig_prediction + fig_ground_truth + fig_diff:
                plt.close(fig)
            plt.close(fig_pca)
        self.validation_step_outputs.clear()

    def test_step(self, graph: DenseGraphBatch, batch_idx: int) -> None:
        graph_emb, graph_pred, _, _ = self(graph=graph)
        outputs = {
            "prediction": graph_pred,
            "ground_truth": graph,
            "graph_emb": graph_emb,
        }
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        if self.trainer.is_global_zero and wandb.run is not None:
            n_examples = 10
            predictions = self.test_step_outputs[0]["prediction"].node_features
            ground_truths = self.test_step_outputs[0]["ground_truth"].node_features
            batch_size = predictions.shape[0]
            n_show = min(n_examples, batch_size)

            graph_emb = torch.cat([el["graph_emb"] for el in self.test_step_outputs], dim=0)
            targets = np.concatenate(
                [el["ground_truth"].y.numpy() for el in self.test_step_outputs], axis=0
            )

            pred_imgs = predictions[:n_show, :, :].detach().cpu()
            gt_imgs = ground_truths[:n_show, :, :].detach().cpu()

            pca_predictions = graph_emb.detach().cpu().float().numpy()
            fig_pca = pL.plot_pca(pca_predictions, targets, n_rows=100, n_cols=8)
            fig_prediction = pL.plot_feature_map(pred_imgs, n_show)
            fig_ground_truth = pL.plot_feature_map(gt_imgs, n_show)

            wandb.log({
                "Test/Prediction": [wandb.Image(fig) for fig in fig_prediction],
                "Test/Ground Truth": [wandb.Image(fig) for fig in fig_ground_truth],
                "Test/PCA": wandb.Image(fig_pca, caption="PCA"),
            })
            for fig in fig_prediction + fig_ground_truth:
                plt.close(fig)
            plt.close(fig_pca)
        self.test_step_outputs.clear()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate),
        validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust
        something about them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.graph_ae = torch.compile(self.graph_ae)

    def configure_optimizers(self) -> tuple:
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you'd need one. But in the case of GANs or similar you
        might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        # Standard transformer split: decay 2-D weight matrices; never decay
        # 1-D parameters (norm gains / biases) or small-init output projections.
        no_decay_names = {"bias", "summary_node", "perm_node"}
        decay_params, no_decay_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or any(nd in name for nd in no_decay_names):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optimizer = self.hparams.optimizer(params=[
            {"params": decay_params},
            {"params": no_decay_params, "weight_decay": 0.0},
        ])
        # Calculate total optimizer steps accounting for dynamic grad accumulation
        # Lightning's estimated_stepping_batches assumes fixed accumulation; here we
        # integrate the configured GradientAccumulationScheduler schedule to avoid
        # getting stuck in warmup.
        try:
            from lightning.pytorch.callbacks import (
                GradientAccumulationScheduler,
            )

            grad_accum_scheduler_cls = GradientAccumulationScheduler
        except Exception:
            grad_accum_scheduler_cls = None

        # Guard: in early setup, Lightning can report `inf` for num_training_batches.
        # If so, fall back to `estimated_stepping_batches` (finite) and skip
        # manual accumulation-aware computation.
        try:
            train_batches_per_epoch = int(self.trainer.num_training_batches)
        except Exception:
            train_batches_per_epoch = None
        max_epochs = int(self.trainer.max_epochs)

        # Default: fixed accumulate_grad_batches from trainer/strategy
        default_accum = int(getattr(self.trainer, "accumulate_grad_batches", 1)) or 1
        schedule = None
        for cb in self.trainer.callbacks:
            if grad_accum_scheduler_cls is not None and isinstance(cb, grad_accum_scheduler_cls):
                schedule = dict(cb.scheduling)
                break

        def accum_for_epoch(epoch: int) -> int:
            if not schedule:
                return default_accum
            # Find the largest milestone <= epoch
            milestone = max([k for k in schedule.keys() if k <= epoch], default=0)
            return int(schedule.get(milestone, default_accum)) or 1

        if train_batches_per_epoch is None or train_batches_per_epoch == float("inf"):
            # Fallback: use Lightning's estimate (already finite and accounts for many trainer flags)
            num_training_steps = int(self.trainer.estimated_stepping_batches)
        else:
            # Optionally simulate a different total number of epochs for the LR schedule
            # This allows "long-run" schedules while actually training fewer epochs.
            simulate_epochs = bool(getattr(self.hparams.scheduler, "simulate_epochs", False))
            simulated_max_epochs = int(
                getattr(self.hparams.scheduler, "simulated_max_epochs", max_epochs)
            )
            effective_epochs = simulated_max_epochs if simulate_epochs else max_epochs

            total_steps = 0
            for epoch in range(effective_epochs):
                accum = accum_for_epoch(epoch)
                # Number of optimizer steps this epoch
                steps_this_epoch = (train_batches_per_epoch + accum - 1) // accum
                total_steps += steps_this_epoch

            # Respect an explicit max_steps if provided
            max_steps = getattr(self.trainer, "max_steps", None)
            if isinstance(max_steps, int) and max_steps > 0:
                num_training_steps = min(total_steps, max_steps)
            else:
                num_training_steps = total_steps
        scheduler_type = getattr(self.hparams.scheduler, "type", "cosine_warmup")

        if scheduler_type == "one_cycle":
            lr_scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hparams.scheduler.max_lr,
                total_steps=num_training_steps,
                pct_start=self.hparams.scheduler.warmup,
                div_factor=self.hparams.scheduler.div_factor,
                final_div_factor=self.hparams.scheduler.final_div_factor,
                anneal_strategy="cos",
            )

            scheduler = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "OneCycleLR",
            }

        else:
            num_warmup_steps = int(
                self.hparams.scheduler.warmup * num_training_steps
            )  # 10% warmup is a common choice

            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            # Step scheduler every batch
            scheduler = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "CosineWarmupLR",
            }
        return [optimizer], [scheduler]

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        ae = self.graph_ae
        component_max_norm = 5.0
        for component in (
            ae.encoder, ae.decoder,
            ae.bottle_neck_encoder, ae.bottle_neck_decoder,
        ):
            torch.nn.utils.clip_grad_norm_(component.parameters(), max_norm=component_max_norm)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Callable[[], None],
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def predict_step(self, batch: DenseGraphBatch, batch_idx: int) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            graph_emb, *_ = self(graph=batch)
            return graph_emb


if __name__ == "__main__":
    pass
