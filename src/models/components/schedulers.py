import torch

from omegaconf import DictConfig


class TemperatureScheduler(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.initial_tau = hparams.initial_tau
        self.final_tau = hparams.final_tau
        self.num_epochs = hparams.num_epochs

    def forward(self, epoch: int) -> float:
        # Exponential decay
        tau = self.initial_tau * (self.final_tau / self.initial_tau) ** (
            epoch / self.num_epochs
        )
        tau = max(self.final_tau, tau)
        return tau


class EntropyWeightScheduler(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.initial_weight = hparams.initial_weight
        self.final_weight = hparams.final_weight
        self.total_epochs = hparams.num_epochs
        self.mode = hparams.mode

    def forward(self, epoch: int) -> float:
        t = min(epoch, self.total_epochs)
        if self.mode == "linear":
            return self.initial_weight * (
                1 - t / self.total_epochs
            ) + self.final_weight * (t / self.total_epochs)
        elif self.mode == "exponential":
            ratio = self.final_weight / self.initial_weight
            return self.initial_weight * (ratio ** (t / self.total_epochs))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
