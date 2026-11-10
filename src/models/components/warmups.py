"""Learning-rate warmup schedules."""

import math

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int
) -> LambdaLR:
    """Linear warmup to the base LR over ``num_warmup_steps``, then cosine decay to zero."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
