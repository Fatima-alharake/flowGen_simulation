from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = None) -> float:
    if data_range is None:
        data_range = float((target.max() - target.min()).item())
        if data_range < 1e-6:
            data_range = 1.0
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-12:
        return 100.0
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    data_range: float = None,
) -> float:
    if data_range is None:
        data_range = float((target.max() - target.min()).item())
        if data_range < 1e-6:
            data_range = 1.0

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    channels = pred.shape[1]
    kernel = torch.ones((channels, 1, window_size, window_size), device=pred.device) / (window_size**2)

    mu_x = F.conv2d(pred, kernel, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=window_size // 2, groups=channels)

    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, kernel, padding=window_size // 2, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=channels) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return float(ssim_map.mean().item())


def compute_field_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    diff = pred - target
    mse = torch.mean(diff**2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(diff)).item()

    norm_target = torch.norm(target.reshape(target.shape[0], -1), p=2, dim=1)
    norm_diff = torch.norm(diff.reshape(diff.shape[0], -1), p=2, dim=1)
    rel_l2 = torch.mean(norm_diff / (norm_target + 1e-7)).item()

    target_std = target.std().item()
    nrmse = rmse / (target_std + 1e-7)

    psnr = compute_psnr(pred, target)
    ssim = compute_ssim(pred, target)

    return {
        "rmse": float(rmse),
        "nrmse": float(nrmse),
        "mae": float(mae),
        "rel_l2": float(rel_l2),
        "psnr": float(psnr),
        "ssim": float(ssim),
    }
