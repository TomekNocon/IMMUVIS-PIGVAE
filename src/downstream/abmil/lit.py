import torch
import pytorch_lightning as pl

from src.downstream.abmil.model import GatedABMIL


class AbmilLitModule(pl.LightningModule):
    """Lightning wrapper around `GatedABMIL` for the downstream MIL classifier.

    Loss is `BCEWithLogitsLoss` for the binary case (`num_classes == 2`, matching
    the single-logit head produced by `GatedABMIL`), otherwise `CrossEntropyLoss`.
    """

    def __init__(self, emb_dim, hidden_dim, num_heads=1, num_classes=2, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = GatedABMIL(emb_dim, hidden_dim, num_heads, num_classes)
        self.num_classes = num_classes
        self.lr = lr
        self.loss = torch.nn.BCEWithLogitsLoss() if num_classes == 2 else torch.nn.CrossEntropyLoss()

    def _step(self, batch, stage):
        bags, mask, y = batch
        logits, _ = self.model(bags, mask)
        target = y.unsqueeze(1).float() if self.num_classes == 2 else y
        loss = self.loss(logits, target)
        self.log(f"{stage}_loss", loss, prog_bar=False, batch_size=bags.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
