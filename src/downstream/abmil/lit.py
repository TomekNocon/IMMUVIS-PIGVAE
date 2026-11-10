"""LightningModule wrapper of the gated ABMIL classifier."""

import lightning as L
import torch

from src.downstream.abmil.model import GatedABMIL


class AbmilLitModule(L.LightningModule):
    """Trains :class:`GatedABMIL` on padded bags.

    The loss is ``BCEWithLogitsLoss`` for the binary case (single-logit head), otherwise
    ``CrossEntropyLoss``.
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        num_heads: int = 1,
        num_classes: int = 2,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GatedABMIL(emb_dim, hidden_dim, num_heads, num_classes)
        self.num_classes = num_classes
        self.lr = lr
        self.loss = (
            torch.nn.BCEWithLogitsLoss() if num_classes == 2 else torch.nn.CrossEntropyLoss()
        )

    def _step(self, batch: tuple[torch.Tensor, ...], stage: str) -> torch.Tensor:
        bags, mask, y = batch
        logits, _ = self.model(bags, mask)
        target = y.unsqueeze(1).float() if self.num_classes == 2 else y
        loss = self.loss(logits, target)
        self.log(f"{stage}_loss", loss, prog_bar=False, batch_size=bags.size(0))
        return loss

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
