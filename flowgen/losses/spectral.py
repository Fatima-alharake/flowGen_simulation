import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    def __init__(self, log_magnitude: bool = True, eps: float = 1e-6):
        super().__init__()
        self.log_magnitude = log_magnitude
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fft_pred = torch.fft.rfft2(pred, norm="ortho")
        fft_target = torch.fft.rfft2(target, norm="ortho")

        mag_pred = torch.abs(fft_pred)
        mag_target = torch.abs(fft_target)

        if self.log_magnitude:
            loss = F.mse_loss(torch.log(mag_pred + self.eps), torch.log(mag_target + self.eps))
        else:
            loss = F.mse_loss(mag_pred, mag_target)

        return loss
