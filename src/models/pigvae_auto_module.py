from typing import Tuple

import torch
import lightning as L
from models.components.warmups import get_cosine_schedule_with_warmup
from models.components.plot import restore_tensor, plot_images
import wandb

import os
import rootutils
rootutils.setup_root(os.getcwd(), indicator=".project-root", pythonpath=True)


# https://stackoverflow.com/questions/65807601/output-prediction-of-pytorch-lightning-model

DATASET_LEN = 2_000


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
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["graph_ae"])
        self.save_hyperparameters(ignore=["critic"])
        self.save_hyperparameters(logger=False)
        self.graph_ae = graph_ae
        self.critic = critic
        self.automatic_optimization = True
        self.validation_step_outputs = []
        self.perms = []

    def forward(self, graph, training):
        graph_pred, perm, mu, logvar = self.graph_ae(graph, training)
        return graph_pred, perm, mu, logvar

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        pass

    def training_step(
        self, graph: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=True,
        )
        loss = self.critic(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
        )
        self.log_dict(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, graph: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=True,
        )
        self.perms.append(perm[:5, :, :])
        outputs = {"prediction": graph_pred, "ground_truth": graph}
        self.validation_step_outputs.append(outputs)
        batch_size = graph_pred.node_features.shape[0]

        metrics_soft = self.critic.evaluate(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
            prefix="val",
        )
        graph_pred, perm, mu, logvar = self(
            graph=graph,
            training=False,
        )
        metrics_hard = self.critic.evaluate(
            graph_true=graph,
            graph_pred=graph_pred,
            perm=perm,
            mu=mu,
            logvar=logvar,
            prefix="val_hard",
        )
        metrics = {**metrics_soft, **metrics_hard}
        self.log_dict(
            metrics,
            sync_dist=False,
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
        )
        return metrics

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # Log one example to W&B
        pred = self.validation_step_outputs[0]["prediction"].node_features
        gt = self.validation_step_outputs[0]["ground_truth"].node_features
        batch_size = pred.shape[0]
        pred_img = (
            restore_tensor(pred, batch_size, 1, 24, 24, 4)
            .detach()
            .cpu()
            .squeeze()
            .numpy()
        )
        gt_img = (
            restore_tensor(gt, batch_size, 1, 24, 24, 4)
            .detach()
            .cpu()
            .squeeze()
            .numpy()
        )
        # p = (self.perms[0].detach()
        #     .cpu()
        #     .squeeze()
        #     .numpy()
        # )
        p = (
            torch.concat([self.perms[0], self.perms[1]], dim= 0)
            .detach()
            .cpu()
            .squeeze()
            .numpy()
            )
        wandb.log(
            {
                "Prediction": wandb.Image(
                    plot_images(pred_img, n_images=10), caption="Predicted Image"
                ),
                "Ground Truth": wandb.Image(
                    plot_images(gt_img, n_images=10), caption="Ground Truth"
                ),
                "Perms": wandb.Image(
                    plot_images(p, n_images=10), caption="Perms"
                ),
            }
        )
        self.validation_step_outputs.clear()
        self.perms.clear()

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # if self.hparams.compile and stage == "fit":
        #     self.net = torch.compile(self.net)
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        # Calculate total training steps based on the actual number of batches
        # This is more accurate than using a fixed DATASET_LEN
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup is a common choice

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        # Step scheduler every batch
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()


if __name__ == "__main__":
    _ = PLGraphAE(None, None)
