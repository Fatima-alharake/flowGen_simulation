from .physical import VorticityLoss, DivergenceLoss, GradientLoss
from .spectral import SpectralLoss
from .objectives import CompositeLoss, gaussian_kl_divergence

__all__ = [
    "VorticityLoss",
    "DivergenceLoss",
    "GradientLoss",
    "SpectralLoss",
    "CompositeLoss",
    "gaussian_kl_divergence",
]
