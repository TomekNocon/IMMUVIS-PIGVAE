"""Offline model-inspection diagnostics (weights, activations, attention, latent, reconstruction)."""

from src.utils.inspection.activations import collect_activation_stats
from src.utils.inspection.attention import attention_entropy_from_input
from src.utils.inspection.latent import (
    ae_vae_spectrum,
    film_diagnostics,
    latent_diagnostics,
)
from src.utils.inspection.reconstruction import (
    image_space_reconstruction,
    reconstruction_diagnostics,
)
from src.utils.inspection.report import write_report
from src.utils.inspection.weights import weight_diagnostics

__all__ = [
    "ae_vae_spectrum",
    "attention_entropy_from_input",
    "collect_activation_stats",
    "film_diagnostics",
    "image_space_reconstruction",
    "latent_diagnostics",
    "reconstruction_diagnostics",
    "weight_diagnostics",
    "write_report",
]
