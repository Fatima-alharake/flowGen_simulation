from typing import Dict
import numpy as np
import torch
from scipy import stats


def compute_uncertainty_metrics(
    samples: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.05,
) -> Dict[str, float]:
    pred_mean = samples.mean(dim=1)
    pred_std = samples.std(dim=1) + 1e-6

    var = pred_std**2
    diff_sq = (target - pred_mean) ** 2
    nll = 0.5 * (torch.log(2.0 * np.pi * var) + diff_sq / var).mean().item()

    z_score = stats.norm.ppf(1.0 - alpha / 2.0)
    lower = pred_mean - z_score * pred_std
    upper = pred_mean + z_score * pred_std

    covered = (target >= lower) & (target <= upper)
    picp = covered.float().mean().item()
    mpiw = (upper - lower).mean().item()

    abs_err = torch.abs(target - pred_mean).reshape(-1)
    std_flat = pred_std.reshape(-1)
    vx = abs_err - torch.mean(abs_err)
    vy = std_flat - torch.mean(std_flat)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2)) + 1e-7)

    spread_to_error_ratio = (pred_std.mean() / (abs_err.mean() + 1e-7)).item()

    return {
        "nll": float(nll),
        "picp": float(picp),
        "mpiw": float(mpiw),
        "nominal_coverage": float(1.0 - alpha),
        "error_spread_correlation": float(corr.item()),
        "spread_to_error_ratio": float(spread_to_error_ratio),
    }
