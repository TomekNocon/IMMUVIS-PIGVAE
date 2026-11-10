"""Custom OmegaConf resolvers used by the Hydra configs."""

from omegaconf import OmegaConf


def register_resolvers() -> None:
    """Register ``${multiply:a,b}`` and ``${divide:a,b}`` (integer arithmetic); idempotent."""
    OmegaConf.register_new_resolver("multiply", lambda x, y: int(x) * int(y), replace=True)
    OmegaConf.register_new_resolver("divide", lambda x, y: int(x) // int(y), replace=True)
