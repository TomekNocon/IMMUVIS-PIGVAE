from __future__ import annotations

import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback


class PermutationDiagnosticsCallback(Callback):
    """Logs permuter health metrics that diagnose why the permuter fails to converge.

    Three questions answered every validation epoch:
      1. grad_norm  — is the permuter actually receiving gradients?
      2. slot_*_consistency — for augmentation slot i, do all images agree on the class?
      3. is_bijective — are all 8 slots mapped to distinct classes?

    If grad_norm == 0 during the freeze window → gradient is blocked somewhere.
    If consistency is low → permuter is inconsistent across images for the same orientation.
    If is_bijective == 0 → Sinkhorn is not enforcing the bijection (bug).
    """

    AUG_NAMES = [
        "r0_f", "r0_nf", "r180_f", "r180_nf",
        "r270_f", "r270_nf", "r90_f", "r90_nf",
    ]

    @staticmethod
    def _grad_norm(module: torch.nn.Module) -> float:
        total_sq = sum(
            p.grad.data.norm(2).item() ** 2
            for p in module.parameters()
            if p.grad is not None
        )
        return total_sq ** 0.5

    def on_after_backward(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        if trainer.global_step % max(trainer.log_every_n_steps, 1) != 0:
            return

        ae = pl_module.graph_ae
        pl_module.log("perm_diag/permuter_grad_norm", self._grad_norm(ae.permuter), on_step=True, on_epoch=False, prog_bar=False)
        pl_module.log("perm_diag/encoder_grad_norm",  self._grad_norm(ae.encoder),  on_step=True, on_epoch=False, prog_bar=False)
        pl_module.log("perm_diag/decoder_grad_norm",  self._grad_norm(ae.decoder),  on_step=True, on_epoch=False, prog_bar=False)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        if not pl_module.validation_step_outputs:
            return

        soft_probs = pl_module.validation_step_outputs[0].get("soft_probs")
        if soft_probs is None:
            return

        num_views = 8
        total_batch = soft_probs.shape[0]
        B = total_batch // num_views

        # preds[slot, img] = predicted class for that slot/image pair
        preds = soft_probs.argmax(dim=-1).view(num_views, B)  # [8, B]

        mode_classes = []
        for slot_idx in range(num_views):
            slot_preds = preds[slot_idx]  # [B]
            mode_class = int(slot_preds.mode().values.item())
            mode_classes.append(mode_class)
            consistency = (slot_preds == mode_class).float().mean().item()
            confidence = (
                soft_probs[slot_idx * B : (slot_idx + 1) * B]
                .max(dim=-1)
                .values.mean()
                .item()
            )
            name = self.AUG_NAMES[slot_idx]
            pl_module.log(f"perm_diag/{name}_class", float(mode_class), on_epoch=True, on_step=False, batch_size=1)
            pl_module.log(f"perm_diag/{name}_consistency", consistency, on_epoch=True, on_step=False, batch_size=1)
            pl_module.log(f"perm_diag/{name}_confidence", confidence, on_epoch=True, on_step=False, batch_size=1)

        # Bijection: all 8 slots should map to distinct classes
        is_bijective = float(len(set(mode_classes)) == num_views)
        pl_module.log("perm_diag/is_bijective", is_bijective, on_epoch=True, on_step=False, batch_size=1)

        # Mean confidence across all samples
        pl_module.log(
            "perm_diag/mean_confidence",
            soft_probs.max(dim=-1).values.mean().item(),
            on_epoch=True,
            on_step=False,
            batch_size=1,
        )

        # Encoder node-feature diversity across augmented views.
        # High aug_cos_sim  → encoder produces similar features for all augmentations
        #                    → permuter is blind (bad).
        # Low  aug_cos_sim  → features are distinctive per augmentation (good).
        # High aug_feat_std → features vary across augmentations (good).
        node_features = pl_module.validation_step_outputs[0].get("node_features")
        if node_features is not None and total_batch % num_views == 0:
            nf = node_features  # [8B, N, D]
            _, N, D = nf.shape
            views = nf.view(num_views, B, N, D)            # [8, B, N, D]
            views_norm = F.normalize(views, dim=-1)
            # Compare same grid position across augmentation pairs (no mean pooling)
            sim = torch.einsum("vbnd,ubnd->uvb", views_norm, views_norm) / N  # [8, 8, B]
            off_diag = ~torch.eye(num_views, device=sim.device, dtype=torch.bool)
            cos_sim = sim[off_diag].mean().item()

            std_across_aug = nf.view(num_views, B, N, D).std(dim=0).mean().item()

            pl_module.log("enc_diag/aug_cos_sim",   cos_sim,         on_epoch=True, on_step=False, batch_size=1)
            pl_module.log("enc_diag/aug_feat_std",  std_across_aug,  on_epoch=True, on_step=False, batch_size=1)
