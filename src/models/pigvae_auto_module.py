from typing import Tuple, Dict, Any, Callable

import torch
import numpy as np
import lightning as L
import matplotlib.pyplot as plt
from collections import Counter
from src.models.components.warmups import get_cosine_schedule_with_warmup
import src.models.components.plot as pL

import src.models.components.metrics.recontructions as R

import wandb
from src.data.components.graphs_datamodules import DenseGraphBatch

import os
import rootutils

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
        temperature_scheduler: torch.nn.Module,
        entropy_weight_scheduler: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        compile: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["graph_ae"])
        self.save_hyperparameters(ignore=["critic"])
        self.save_hyperparameters(ignore=["temperature_scheduler"])
        self.save_hyperparameters(ignore=["entropy_weight_scheduler"])
        self.save_hyperparameters(logger=False)
        self.graph_ae = graph_ae
        self.critic = critic
        self.temperature_scheduler = temperature_scheduler
        self.entropy_weight_scheduler = entropy_weight_scheduler
        self.automatic_optimization = True
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.perms = []

    def forward(self, graph: DenseGraphBatch, training: bool, tau: float) -> Tuple:
        graph_emb, graph_pred, soft_probs, perm, mu, logvar = self.graph_ae(
            graph, training, tau
        )
        return graph_emb, graph_pred, soft_probs, perm, mu, logvar

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        pass

    def training_step(self, graph: DenseGraphBatch, batch_idx: int) -> torch.Tensor:
        tau = self.temperature_scheduler(self.current_epoch)
        beta = self.entropy_weight_scheduler(self.current_epoch)
        graph_emb, graph_pred, soft_probs, perm, mu, logvar = self(
            graph=graph, training=True, tau=tau
        )
        loss = self.critic(
            graph_emb=graph_emb,
            graph_true=graph,
            graph_pred=graph_pred,
            soft_probs=soft_probs,
            perm=perm,
            beta=beta,
            mu=mu,
            logvar=logvar,
        )
        self.log_dict(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    # def on_after_backward(self) -> None:
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             grad_norm = torch.norm(param.grad)
    #             if torch.isnan(grad_norm) or grad_norm > 1e3:
    #                 print(f"🚨 Problematic gradient: {name} = {grad_norm}")

    def validation_step(self, graph: DenseGraphBatch, batch_idx: int) -> Dict[str, Any]:
        tau = self.temperature_scheduler(self.current_epoch)
        beta = self.entropy_weight_scheduler(self.current_epoch)
        graph_emb, graph_pred, soft_probs, perm, mu, logvar = self(
            graph=graph, training=True, tau=tau
        )

        if perm is not None:
            self.perms.append(perm)
        outputs = {
            "prediction": graph_pred,
            "ground_truth": graph,
            "graph_emb": graph_emb,
            "soft_probs": soft_probs,
        }
        self.validation_step_outputs.append(outputs)
        batch_size = graph_pred.node_features.shape[0]

        metrics_soft = self.critic.evaluate(
            graph_emb=graph_emb,
            graph_true=graph,
            graph_pred=graph_pred,
            soft_probs=soft_probs,
            perm=perm,
            beta=beta,
            mu=mu,
            logvar=logvar,
            prefix="val",
        )
        graph_emb, graph_pred, soft_probs, perm, mu, logvar = self(
            graph=graph, training=False, tau=1.0
        )
        metrics_hard = self.critic.evaluate(
            graph_emb=graph_emb,
            graph_true=graph,
            graph_pred=graph_pred,
            soft_probs=soft_probs,
            perm=perm,
            beta=0.0,
            mu=mu,
            logvar=logvar,
            prefix="val_hard",
        )
        # sample_graph = graph.take_sample(16)
        # lie_metrics = lE.get_equivariance_metrics(self, sample_graph)
        metrics = {
            **metrics_soft,
            **metrics_hard,
            # **lie_metrics,
            "tau": tau,
            "beta": beta,
        }
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
        n_examples = 10
        predictions = self.validation_step_outputs[0]["prediction"].node_features
        ground_truths = self.validation_step_outputs[0]["ground_truth"].node_features
        argsort = self.validation_step_outputs[0][
            "ground_truth"
        ].argsort_augmented_features
        graph_emb = self.validation_step_outputs[0]["graph_emb"]
        targets = self.validation_step_outputs[0]["ground_truth"].y
        soft_probs = self.validation_step_outputs[0]["soft_probs"]

        if soft_probs is not None:
            perm_preds = torch.argmax(soft_probs, dim=1).detach().cpu().numpy().tolist()
            perm_preds_counter = Counter(perm_preds)
            fig_perm_preds_counter = pL.plot_barchart_from_dict(
                dict(perm_preds_counter), "Perm Preds Counter"
            )

        batch_size = predictions.shape[0] // 8
        idx_to_show = R.batch_augmented_indices(
            batch_size, num_permutations=8, n_examples=n_examples
        )
        if self.perms:
            perms = self.perms[0]
            subset_perms = perms[idx_to_show, :, :]
            permutations = subset_perms.detach().cpu().squeeze().numpy()
        else:
            permutations = np.array([])
        subset_predictions = predictions[idx_to_show, :, :]
        subset_targets = targets[:n_examples].to(torch.int)
        subset_ground_truths = ground_truths[idx_to_show, :, :]
        subset_graph_emb = graph_emb[idx_to_show, :]
        subset_argsort = argsort[idx_to_show, :]

        restore_subset_predictions = torch.stack(
            [img[arg, :] for img, arg in zip(subset_predictions, subset_argsort)], dim=0
        )

        restore_subset_ground_truths = torch.stack(
            [img[arg, :] for img, arg in zip(subset_ground_truths, subset_argsort)],
            dim=0,
        )

        subset_batch_size = subset_predictions.shape[0]
        pred_imgs = (
            pL.restore_tensor(
                restore_subset_predictions, subset_batch_size, 1, 24, 24, 4
            )
            .detach()
            .cpu()
            .squeeze()
        )
        ground_truth_imgs = (
            pL.restore_tensor(
                restore_subset_ground_truths, subset_batch_size, 1, 24, 24, 4
            )
            .detach()
            .cpu()
            .squeeze()
        )
        pca_predictions = subset_graph_emb.detach().cpu().squeeze().numpy()

        mse = R.mse_per_transform(ground_truth_imgs, pred_imgs, n_examples, 8)
        fig_prediction = pL.plot_images_all_perm(
            pred_imgs.numpy(), n_rows=n_examples, n_cols=8
        )
        fig_ground_truth = pL.plot_images_all_perm(
            ground_truth_imgs.numpy(), n_rows=n_examples, n_cols=8
        )
        fig_perms = pL.plot_images_all_perm(permutations, n_rows=n_examples, n_cols=8)
        fig_pca = pL.plot_pca(
            pca_predictions, subset_targets, n_rows=n_examples, n_cols=8
        )
        fig_mse = pL.plot_barchart_from_dict(mse, "MSE per transform")

        best_k, fig_silhouette = pL.plot_silhouette(pca_predictions)
        fig_inter = pL.plot_inter_silhouette(pca_predictions, best_k)
        wandb.log(
            {
                "Prediction": wandb.Image(fig_prediction, caption="Predicted Image"),
                "Ground Truth": wandb.Image(fig_ground_truth, caption="Ground Truth"),
                "Perms": wandb.Image(fig_perms, caption="Perms")
                if self.perms
                else None,
                "PCA": wandb.Image(fig_pca, caption="PCA"),
                "MSE Per Transform": wandb.Image(fig_mse, caption="MSE"),
                "Perm Preds Counter": wandb.Image(
                    fig_perm_preds_counter, caption="Perm Preds Counter"
                )
                if soft_probs is not None
                else None,
                "Silhouette": wandb.Image(fig_silhouette, caption="Silhouette"),
                "Inter Silhouette": wandb.Image(fig_inter, caption="Inter Silhouette"),
            }
        )
        plt.close(fig_prediction)
        plt.close(fig_ground_truth)
        plt.close(fig_perms)
        plt.close(fig_pca)
        plt.close(fig_mse)
        if soft_probs is not None:
            plt.close(fig_perm_preds_counter)
        plt.close(fig_silhouette)
        plt.close(fig_inter)
        self.validation_step_outputs.clear()
        self.perms.clear()

    def test_step(
        self, graph: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        graph_emb, graph_pred, _, perm, _, _ = self(
            graph=graph, training=False, tau=1.0
        )
        if perm is not None:
            self.perms.append(perm)
        outputs = {
            "prediction": graph_pred,
            "ground_truth": graph,
            "graph_emb": graph_emb,
        }
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        n_examples = 10
        predictions = self.test_step_outputs[0]["prediction"].node_features
        ground_truths = self.test_step_outputs[0]["ground_truth"].node_features
        argsort = self.test_step_outputs[0]["ground_truth"].argsort_augmented_features
        graph_emb = torch.cat([el["graph_emb"] for el in self.test_step_outputs], dim=0)
        targets = np.concatenate(
            [el["ground_truth"].y for el in self.test_step_outputs], axis=0
        )
        batch_size = predictions.shape[0] // 8
        idx_to_show = R.batch_augmented_indices(
            batch_size, num_permutations=8, n_examples=n_examples
        )
        if self.perms:
            perms = self.perms[0]
            subset_perms = perms[idx_to_show, :, :]
            permutations = subset_perms.detach().cpu().squeeze().numpy()
        else:
            permutations = np.array([])
        subset_predictions = predictions[idx_to_show, :, :]
        subset_ground_truths = ground_truths[idx_to_show, :, :]
        subset_argsort = argsort[idx_to_show, :]

        restore_subset_predictions = torch.stack(
            [img[arg, :] for img, arg in zip(subset_predictions, subset_argsort)], dim=0
        )

        restore_subset_ground_truths = torch.stack(
            [img[arg, :] for img, arg in zip(subset_ground_truths, subset_argsort)],
            dim=0,
        )

        subset_batch_size = subset_predictions.shape[0]
        pred_imgs = (
            pL.restore_tensor(
                restore_subset_predictions, subset_batch_size, 1, 24, 24, 4
            )
            .detach()
            .cpu()
            .squeeze()
            .numpy()
        )
        ground_truth_imgs = (
            pL.restore_tensor(
                restore_subset_ground_truths, subset_batch_size, 1, 24, 24, 4
            )
            .detach()
            .cpu()
            .squeeze()
            .numpy()
        )
        pca_predictions = graph_emb.detach().cpu().squeeze().numpy()
        fig_prediction = pL.plot_images_all_perm(pred_imgs, n_rows=n_examples, n_cols=8)
        fig_ground_truth = pL.plot_images_all_perm(
            ground_truth_imgs, n_rows=n_examples, n_cols=8
        )
        fig_perms = pL.plot_images_all_perm(permutations, n_rows=n_examples, n_cols=8)
        fig_pca = pL.plot_pca(pca_predictions, targets, n_rows=100, n_cols=8)
        wandb.log(
            {
                "Prediction": wandb.Image(fig_prediction, caption="Predicted Image"),
                "Ground Truth": wandb.Image(fig_ground_truth, caption="Ground Truth"),
                "Perms": wandb.Image(fig_perms, caption="Perms")
                if self.perms
                else None,
                "PCA": wandb.Image(fig_pca, caption="PCA"),
            }
        )
        plt.close(fig_prediction)
        plt.close(fig_ground_truth)
        plt.close(fig_perms)
        plt.close(fig_pca)
        self.test_step_outputs.clear()
        self.perms.clear()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.graph_ae = torch.compile(self.graph_ae)

    def configure_optimizers(self) -> Tuple:
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
        num_warmup_steps = int(
            self.hparams.scheduler.warmup * num_training_steps
        )  # 10% warmup is a common choice

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        # Step scheduler every batch
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

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
            tau = 1.0  # or any fixed temperature you want at inference
            graph_emb, *_ = self(graph=batch, training=False, tau=tau)
            return graph_emb


if __name__ == "__main__":
    _ = PLGraphAE(None, None)
