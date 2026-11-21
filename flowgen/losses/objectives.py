from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .physical import VorticityLoss, DivergenceLoss, GradientLoss
from .spectral import SpectralLoss


def gaussian_kl_divergence(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p)**2) / (var_p + 1e-8) - 1.0)
    return torch.mean(torch.sum(kl, dim=(1, 2, 3)))


class CompositeLoss(nn.Module):
    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 0.001,
        vorticity_weight: float = 0.0,
        divergence_weight: float = 0.0,
        spectral_weight: float = 0.0,
        gradient_weight: float = 0.0,
        recon_type: str = "l1",
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.vorticity_weight = vorticity_weight
        self.divergence_weight = divergence_weight
        self.spectral_weight = spectral_weight
        self.gradient_weight = gradient_weight
        self.recon_type = recon_type

        self.vorticity_fn = VorticityLoss() if vorticity_weight > 0 else None
        self.divergence_fn = DivergenceLoss() if divergence_weight > 0 else None
        self.spectral_fn = SpectralLoss() if spectral_weight > 0 else None
        self.gradient_fn = GradientLoss() if gradient_weight > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        post_mu: torch.Tensor = None,
        post_logvar: torch.Tensor = None,
        prior_mu: torch.Tensor = None,
        prior_logvar: torch.Tensor = None,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        if self.recon_type == "l1":
            recon = F.l1_loss(pred, target)
        elif self.recon_type == "l2":
            recon = F.mse_loss(pred, target)
        elif self.recon_type == "smooth_l1":
            recon = F.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown recon loss type: {self.recon_type}")

        total = self.recon_weight * recon
        losses["recon"] = recon

        if post_mu is not None and prior_mu is not None and self.kl_weight > 0:
            kl = gaussian_kl_divergence(post_mu, post_logvar, prior_mu, prior_logvar)
            scaled_kl = beta * self.kl_weight * kl
            total = total + scaled_kl
            losses["kl"] = kl
            losses["scaled_kl"] = scaled_kl

        if self.vorticity_fn is not None and pred.shape[1] >= 2:
            vort = self.vorticity_fn(pred, target)
            total = total + self.vorticity_weight * vort
            losses["vorticity"] = vort

        if self.divergence_fn is not None and pred.shape[1] >= 2:
            div = self.divergence_fn(pred)
            total = total + self.divergence_weight * div
            losses["divergence"] = div

        if self.spectral_fn is not None:
            spec = self.spectral_fn(pred, target)
            total = total + self.spectral_weight * spec
            losses["spectral"] = spec

        if self.gradient_fn is not None:
            grad = self.gradient_fn(pred, target)
            total = total + self.gradient_weight * grad
            losses["gradient"] = grad

        losses["total"] = total
        return losses
