import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDegradation(nn.Module):
    def __init__(self, scale_factor: int = 4, mode: str = "bicubic", noise_std: float = 0.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "subsample":
            lr = x[..., ::self.scale_factor, ::self.scale_factor]
        elif self.mode == "avgpool":
            lr = F.avg_pool2d(x, kernel_size=self.scale_factor, stride=self.scale_factor)
        elif self.mode in ("bicubic", "bilinear"):
            h, w = x.shape[-2], x.shape[-1]
            target_size = (h // self.scale_factor, w // self.scale_factor)
            align = False if self.mode == "bicubic" else False
            lr = F.interpolate(x, size=target_size, mode=self.mode, align_corners=align)
        else:
            raise ValueError(f"Unsupported degradation mode: {self.mode}")

        if self.noise_std > 0.0 and self.training:
            lr = lr + torch.randn_like(lr) * self.noise_std

        return lr
