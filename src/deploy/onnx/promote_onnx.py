import wandb
import hydra
from omegaconf import DictConfig

from src.utils.pylogger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


def is_better(new: float, old: float, mode: str) -> bool:
    mode = mode.lower()
    if mode == "min":
        return new < old
    if mode == "max":
        return new > old
    raise ValueError("mode must be 'min' or 'max'")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="deploy/onnx/promote.yaml")
def main(cfg: DictConfig) -> None:
    api = wandb.Api()

    candidate = api.artifact(cfg.candidate)
    new_val = float(candidate.metadata[cfg.metric_key])
    log.info(f"Candidate {candidate.name} {cfg.metric_key}={new_val}")

    prod_spec = f"{cfg.entity}/{cfg.project}/{cfg.production_name}:production"
    try:
        current = api.artifact(prod_spec)
        old_val = float(current.metadata[cfg.metric_key])
        log.info(f"Current production {current.name} {cfg.metric_key}={old_val}")
    except wandb.CommError:
        current = None
        old_val = None
        log.warning("No existing production; promoting candidate")

    promote = current is None or is_better(new_val, old_val, cfg.mode)
    if not promote:
        log.info("Candidate worse than production; skip promotion")
        return

    # Add production alias to candidate
    cand_aliases = set(candidate.aliases)
    cand_aliases.add("production")
    candidate.aliases = list(cand_aliases)
    candidate.save()
    log.info(f"Promoted {candidate.name} -> production")

    # Remove production alias from previous
    if current is not None and current.id != candidate.id:
        current.aliases = [a for a in current.aliases if a != "production"]
        current.save()
        log.info(f"Removed production alias from {current.name}")


if __name__ == "__main__":
    main()


