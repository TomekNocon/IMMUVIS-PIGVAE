"""LightningModule that trains PIGVAE on batches of D4-augmented node grids.

Each batch holds ``num_permutations`` (8) stacked view blocks of the same ``B`` samples.
Per-epoch schedules set the permuter temperature, the permutation-loss weight and the KL
annealing factor; the :class:`~src.models.components.model.Critic` computes loss and metrics.
"""

from collections import Counter
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import GradientAccumulationScheduler
from torch.optim.lr_scheduler import OneCycleLR

import src.models.components.plot as pL
import src.models.components.reconstruction_metrics as R
from src.data.components.graphs_datamodules import DenseGraphBatch
from src.models.components.warmups import get_cosine_schedule_with_warmup

NUM_VIEWS = 8  # D4 group size: views stacked per sample in a batch
N_EXAMPLES = 4  # samples shown in the validation figures
COMPONENT_MAX_GRAD_NORM = 5.0


class PLGraphAE(L.LightningModule):
    """PIGVAE training, validation and test loops.

    Args:
        graph_ae: The :class:`~src.models.components.modules.GraphAE`.
        critic: Loss and metrics (:class:`~src.models.components.model.Critic`).
        temperature_scheduler: Epoch -> permuter temperature ``tau``.
        entropy_weight_scheduler: Epoch -> permutation-loss weight.
        kld_alpha_scheduler: Epoch -> KL annealing factor.
        optimizer: Partial optimizer, called with the parameter groups.
        scheduler: LR schedule config (``type``: ``cosine_warmup`` or ``one_cycle``).
        compile: ``torch.compile`` the autoencoder for fitting.
    """

    def __init__(
        self,
        graph_ae: torch.nn.Module,
        critic: torch.nn.Module,
        temperature_scheduler: torch.nn.Module,
        entropy_weight_scheduler: torch.nn.Module,
        kld_alpha_scheduler: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        compile: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "graph_ae",
                "critic",
                "temperature_scheduler",
                "entropy_weight_scheduler",
                "kld_alpha_scheduler",
            ],
            logger=False,
        )
        self.graph_ae = graph_ae
        self.critic = critic
        self.temperature_scheduler = temperature_scheduler
        self.entropy_weight_scheduler = entropy_weight_scheduler
        self.kld_alpha_scheduler = kld_alpha_scheduler
        # The first validation batch (and its permutations) is kept for the epoch-end figures.
        self.validation_step_outputs: list[dict[str, Any]] = []
        self.perms: list[torch.Tensor] = []

    def forward(self, graph: DenseGraphBatch, training: bool, tau: float) -> tuple:
        """``(graph_emb, graph_pred, soft_probs, perm, mu, logvar)``."""
        return self.graph_ae(graph, training, tau)

    # ------------------------------------------------------------------ schedules

    def _schedules(self) -> tuple[float, float, float]:
        """``(tau, permutation-loss weight, KL alpha)`` for the current epoch."""
        epoch = self.current_epoch
        return (
            self.temperature_scheduler(epoch),
            self.entropy_weight_scheduler(epoch),
            self.kld_alpha_scheduler(epoch),
        )

    def _apply_curriculum(self) -> None:
        """Keep the learned permuter off until ``curriculum_epoch``, optionally freezing the
        encoder/decoder for ``freeze_epochs`` afterwards so the permuter can warm up."""
        permuter = self.graph_ae.permuter
        if permuter.curriculum_epoch <= 0:
            return

        permuter.turn_off = self.current_epoch < permuter.curriculum_epoch
        self.log("permuter/turn_off", float(permuter.turn_off), batch_size=1)

        freeze_epochs = getattr(permuter, "freeze_epochs", 0)
        if freeze_epochs > 0:
            start = permuter.curriculum_epoch
            in_freeze_window = start <= self.current_epoch < start + freeze_epochs
            for name, param in self.graph_ae.named_parameters():
                if "permuter" not in name:
                    param.requires_grad = not in_freeze_window
            self.log("permuter/enc_dec_frozen", float(in_freeze_window), batch_size=1)

    # ------------------------------------------------------------------ steps

    def training_step(self, graph: DenseGraphBatch, batch_idx: int) -> dict[str, Any]:
        self._apply_curriculum()
        tau, beta, alpha = self._schedules()
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
            kld_alpha=alpha,
            mu=mu,
            logvar=logvar,
        )
        self.log_dict(loss)
        if mu is not None:
            self._log_latent_stats(mu, logvar, alpha, prefix="")
        return loss

    def _evaluation_step(self, graph: DenseGraphBatch, prefix: str) -> dict[str, Any]:
        """Deterministic forward pass; logs ``{prefix}_*`` metrics averaged over the epoch."""
        self._apply_curriculum()
        tau, beta, alpha = self._schedules()
        graph_emb, graph_pred, soft_probs, perm, mu, logvar = self(
            graph=graph, training=False, tau=tau
        )
        # Keep only the first batch for the epoch-end figures (O(1) memory, not O(N)).
        if prefix == "val":
            if perm is not None and not self.perms:
                self.perms.append(perm)
            if not self.validation_step_outputs:
                self.validation_step_outputs.append({
                    "prediction": graph_pred,
                    "ground_truth": graph,
                    "graph_emb": graph_emb,
                    "soft_probs": soft_probs,
                })

        metrics = self.critic.evaluate(
            graph_emb=graph_emb,
            graph_true=graph,
            graph_pred=graph_pred,
            soft_probs=soft_probs,
            perm=perm,
            beta=beta,
            kld_alpha=alpha,
            mu=mu,
            logvar=logvar,
            prefix=prefix,
        )
        metrics = {**metrics, "tau": tau, "beta": beta, "alpha": alpha}
        self.log_dict(
            metrics,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            batch_size=graph_pred.node_features.shape[0],
        )
        if mu is not None:
            self._log_latent_stats(mu, logvar, alpha, prefix=f"{prefix}_")
        return metrics

    def validation_step(self, graph: DenseGraphBatch, batch_idx: int) -> dict[str, Any]:
        return self._evaluation_step(graph, prefix="val")

    def test_step(self, graph: DenseGraphBatch, batch_idx: int) -> dict[str, Any]:
        return self._evaluation_step(graph, prefix="test")

    def predict_step(self, batch: DenseGraphBatch, batch_idx: int) -> torch.Tensor:
        """Mean-pooled node latent of each graph."""
        graph_emb, *_ = self(graph=batch, training=False, tau=1.0)
        return graph_emb

    def _log_latent_stats(
        self, mu: torch.Tensor, logvar: torch.Tensor, kld_alpha: float, prefix: str = ""
    ) -> None:
        bs = mu.shape[0]
        std = (0.5 * logvar).exp()
        per_dim_kld = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        per_dim_kld_mean = per_dim_kld.mean(0)  # average over the batch
        p = f"{prefix}latent/"
        self.log(f"{p}mu_mean", mu.mean(), batch_size=bs)
        self.log(f"{p}mu_std", mu.std(), batch_size=bs)
        self.log(f"{p}std_mean", std.mean(), batch_size=bs)
        self.log(f"{p}active_dims", (per_dim_kld_mean > 0.1).float().sum(), batch_size=bs)
        self.log(f"{p}kld_per_dim_median", per_dim_kld_mean.median(), batch_size=bs)
        self.log(f"{p}kld_alpha", kld_alpha, batch_size=bs)

    # ------------------------------------------------------------------ figures

    def on_validation_epoch_end(self) -> None:
        if self.trainer.is_global_zero and self.validation_step_outputs and wandb.run is not None:
            self._log_validation_figures(self.validation_step_outputs[0])
        self.validation_step_outputs.clear()
        self.perms.clear()

    def _log_validation_figures(self, outputs: dict[str, Any]) -> None:
        """Log reconstructions, errors, permutations and latent PCA of the first val batch."""
        predictions = outputs["prediction"].node_features
        ground_truth = outputs["ground_truth"]
        soft_probs = outputs["soft_probs"]

        batch_size = predictions.shape[0] // NUM_VIEWS
        idx = R.batch_augmented_indices(batch_size, NUM_VIEWS, n_examples=N_EXAMPLES)
        argsort = ground_truth.argsort_augmented_features[idx, :]

        def restore(nodes: torch.Tensor) -> torch.Tensor:
            """Undo the node shuffle of each view."""
            return (
                torch
                .stack([img[arg, :] for img, arg in zip(nodes[idx, :, :], argsort, strict=True)])
                .detach()
                .cpu()
            )

        pred_imgs = restore(predictions)
        true_imgs = restore(ground_truth.node_features)
        diff = pred_imgs - true_imgs

        vmin = min(pred_imgs.min().item(), true_imgs.min().item())
        vmax = max(pred_imgs.max().item(), true_imgs.max().item())
        diff_max = diff.abs().max().item()

        permutations = (
            self.perms[0][idx, :, :].detach().cpu().squeeze().float().numpy()
            if self.perms
            else np.array([])
        )
        targets = ground_truth.y[:N_EXAMPLES].to(torch.int)
        embeddings = outputs["graph_emb"][idx, :].detach().cpu().squeeze().float().numpy()
        mse = R.mse_per_transform(true_imgs, pred_imgs, N_EXAMPLES, NUM_VIEWS)

        fig_prediction = pL.plot_feature_map(pred_imgs, N_EXAMPLES, vmin=vmin, vmax=vmax)
        fig_ground_truth = pL.plot_feature_map(true_imgs, N_EXAMPLES, vmin=vmin, vmax=vmax)
        fig_diff = pL.plot_feature_map(diff, N_EXAMPLES, vmin=-diff_max, vmax=diff_max)
        fig_perms = pL.plot_images_all_perm(permutations, n_rows=N_EXAMPLES, n_cols=NUM_VIEWS)
        fig_pca = pL.plot_pca(embeddings, targets, n_rows=N_EXAMPLES, n_cols=NUM_VIEWS)
        fig_mse = pL.plot_barchart_from_dict(mse, "MSE per transform")
        fig_counter = None
        if soft_probs is not None:
            counts = Counter(torch.argmax(soft_probs, dim=1).detach().cpu().numpy().tolist())
            fig_counter = pL.plot_barchart_from_dict(dict(counts), "Perm Preds Counter")

        wandb.log({
            "Predictions": [
                wandb.Image(fig, caption=f"Predictions {i + 1}")
                for i, fig in enumerate(fig_prediction)
            ],
            "Ground Truth": [
                wandb.Image(fig, caption=f"Ground Truth {i + 1}")
                for i, fig in enumerate(fig_ground_truth)
            ],
            "Diff": [wandb.Image(fig, caption=f"Diff {i + 1}") for i, fig in enumerate(fig_diff)],
            "Perms": wandb.Image(fig_perms, caption="Perms") if self.perms else None,
            "PCA": wandb.Image(fig_pca, caption="PCA"),
            "MSE Per Transform": wandb.Image(fig_mse, caption="MSE"),
            "Perm Preds Counter": (
                wandb.Image(fig_counter, caption="Perm Preds Counter") if fig_counter else None
            ),
        })
        for fig in [*fig_prediction, *fig_ground_truth, *fig_diff, fig_perms, fig_pca, fig_mse]:
            plt.close(fig)
        if fig_counter is not None:
            plt.close(fig_counter)

    # ------------------------------------------------------------------ optimisation

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.graph_ae = torch.compile(self.graph_ae)

    def _param_groups(self) -> list[dict[str, Any]]:
        """Decay 2-D weights; never decay 1-D params (norm gains, biases) or learned tokens.

        With ``decoder.film_weight_decay`` set, the FiLM projections get their own (heavier)
        weight decay so the gamma/beta generators cannot grow large.
        """
        film_wd = getattr(getattr(self.graph_ae, "decoder", None), "film_weight_decay", None)
        no_decay_names = {"bias", "summary_node", "perm_node"}
        decay, no_decay, film = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if film_wd is not None and "decoder.film" in name and param.ndim > 1:
                film.append(param)
            elif param.ndim == 1 or any(nd in name for nd in no_decay_names):
                no_decay.append(param)
            else:
                decay.append(param)
        groups = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
        if film:
            groups.append({"params": film, "weight_decay": float(film_wd)})
        return groups

    def _num_training_steps(self) -> int:
        """Total optimizer steps, integrating a ``GradientAccumulationScheduler`` if present.

        Lightning's ``estimated_stepping_batches`` assumes a fixed accumulation, which would
        leave the schedule stuck in warmup. ``scheduler.simulate_epochs`` lets the LR schedule
        span ``simulated_max_epochs`` while training fewer epochs.
        """
        try:
            batches_per_epoch = int(self.trainer.num_training_batches)
        except (OverflowError, TypeError, ValueError):  # inf early in setup
            return int(self.trainer.estimated_stepping_batches)

        default_accum = int(getattr(self.trainer, "accumulate_grad_batches", 1)) or 1
        schedule = next(
            (
                dict(cb.scheduling)
                for cb in self.trainer.callbacks
                if isinstance(cb, GradientAccumulationScheduler)
            ),
            None,
        )

        def accum_for_epoch(epoch: int) -> int:
            if not schedule:
                return default_accum
            milestone = max([k for k in schedule if k <= epoch], default=0)
            return int(schedule.get(milestone, default_accum)) or 1

        max_epochs = int(self.trainer.max_epochs)
        if bool(getattr(self.hparams.scheduler, "simulate_epochs", False)):
            max_epochs = int(getattr(self.hparams.scheduler, "simulated_max_epochs", max_epochs))

        total_steps = sum(
            (batches_per_epoch + accum_for_epoch(e) - 1) // accum_for_epoch(e)
            for e in range(max_epochs)
        )
        max_steps = getattr(self.trainer, "max_steps", None)
        if isinstance(max_steps, int) and max_steps > 0:
            return min(total_steps, max_steps)
        return total_steps

    def configure_optimizers(self) -> tuple[list, list]:
        optimizer = self.hparams.optimizer(params=self._param_groups())
        num_training_steps = self._num_training_steps()
        cfg = self.hparams.scheduler

        if getattr(cfg, "type", "cosine_warmup") == "one_cycle":
            lr_scheduler = OneCycleLR(
                optimizer,
                max_lr=cfg.max_lr,
                total_steps=num_training_steps,
                pct_start=cfg.warmup,
                div_factor=cfg.div_factor,
                final_div_factor=cfg.final_div_factor,
                anneal_strategy="cos",
            )
            name = "OneCycleLR"
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(cfg.warmup * num_training_steps),
                num_training_steps=num_training_steps,
            )
            name = "CosineWarmupLR"
        return [optimizer], [
            {"scheduler": lr_scheduler, "interval": "step", "frequency": 1, "name": name}
        ]

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip each component to the same norm budget before the global clip.

        Raw gradient norms reach 1e5 in the encoder/decoder; with only a global clip the
        largest component would consume the whole budget and starve the others.
        """
        ae = self.graph_ae
        components = [ae.encoder, ae.decoder, ae.node_bottleneck]
        if hasattr(ae, "permuter"):
            components.append(ae.permuter)
        for component in components:
            torch.nn.utils.clip_grad_norm_(component.parameters(), max_norm=COMPONENT_MAX_GRAD_NORM)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
