from torch.optim.lr_scheduler import LambdaLR
import math


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Creates a schedule with a learning rate that first increases linearly during the warmup period
    and then decreases following a cosine function.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
