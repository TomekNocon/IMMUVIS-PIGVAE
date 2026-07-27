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
        tau = self.initial_tau * (self.final_tau / self.initial_tau) ** (epoch / self.num_epochs)
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
            return self.initial_weight * (1 - t / self.total_epochs) + self.final_weight * (
                t / self.total_epochs
            )
        elif self.mode == "exponential":
            ratio = self.final_weight / self.initial_weight
            return self.initial_weight * (ratio ** (t / self.total_epochs))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class KLDAlphaScheduler(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.initial_alpha = hparams.initial_alpha
        self.final_alpha = hparams.final_alpha
        self.total_epochs = hparams.num_epochs
        # Hold at initial_alpha until start_epoch, then anneal over num_epochs. Lets the
        # KL ramp start *after* the LR warmup peak so prior pressure and peak LR don't
        # collide (see the ep3 overshoot in vae6_lowkl). Default 0 = ramp from epoch 0.
        self.start_epoch = int(getattr(hparams, "start_epoch", 0))
        self.mode = getattr(hparams, "mode", "linear")

    def forward(self, epoch: int) -> float:
        # epochs since the ramp started, clamped to [0, total_epochs]
        t = min(max(epoch - self.start_epoch, 0), self.total_epochs)
        if self.mode == "linear":
            return self.initial_alpha * (1 - t / self.total_epochs) + self.final_alpha * (
                t / self.total_epochs
            )
        elif self.mode == "exponential":
            # Guard zero initial_alpha in exponential mode
            start = max(self.initial_alpha, 1e-12)
            ratio = self.final_alpha / start
            return start * (ratio ** (t / self.total_epochs))
        elif self.mode == "constant":
            return self.final_alpha
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class LinearWarmupThenConstant(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.start = float(getattr(hparams, "start", 0.0))
        self.end = float(hparams.end)
        self.warmup_epochs = int(hparams.warmup_epochs)

    def forward(self, epoch: int) -> float:
        if epoch >= self.warmup_epochs:
            return self.end
        ratio = max(0.0, float(epoch) / max(1, self.warmup_epochs))
        return self.start * (1.0 - ratio) + self.end * ratio


class LinearDecay(torch.nn.Module):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.start_value = float(hparams.start_value)
        self.end_value = float(hparams.end_value)
        self.start_epoch = int(hparams.start_epoch)
        self.end_epoch = int(hparams.end_epoch)
        if self.end_epoch >= self.start_epoch:
            raise ValueError("end_epoch must be >= start_epoch")

    def forward(self, epoch: int) -> float:
        if epoch <= self.start_epoch:
            return self.start_value
        if epoch >= self.end_epoch:
            return self.end_value
        t = (epoch - self.start_epoch) / max(1, (self.end_epoch - self.start_epoch))
        return self.start_value * (1.0 - t) + self.end_value * t
