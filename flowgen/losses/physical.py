import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_gradients(x: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    b, c, h, w = x.shape
    du_dx = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1)) / (2.0 * dx)
    du_dy = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2)) / (2.0 * dx)
    return du_dx, du_dy


def compute_vorticity(u_field: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    u = u_field[:, 0:1, :, :]
    v = u_field[:, 1:2, :, :]
    _, du_dy = spatial_gradients(u, dx=dx)
    dv_dx, _ = spatial_gradients(v, dx=dx)
    return dv_dx - du_dy


def compute_divergence(u_field: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    u = u_field[:, 0:1, :, :]
    v = u_field[:, 1:2, :, :]
    du_dx, _ = spatial_gradients(u, dx=dx)
    _, dv_dy = spatial_gradients(v, dx=dx)
    return du_dx + dv_dy


class VorticityLoss(nn.Module):
    def __init__(self, dx: float = 1.0, loss_type: str = "l2"):
        super().__init__()
        self.dx = dx
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w_pred = compute_vorticity(pred, dx=self.dx)
        w_target = compute_vorticity(target, dx=self.dx)
        if self.loss_type == "l1":
            return F.l1_loss(w_pred, w_target)
        return F.mse_loss(w_pred, w_target)


class DivergenceLoss(nn.Module):
    def __init__(self, dx: float = 1.0):
        super().__init__()
        self.dx = dx

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        div = compute_divergence(pred, dx=self.dx)
        return torch.mean(div**2)


class GradientLoss(nn.Module):
    def __init__(self, dx: float = 1.0):
        super().__init__()
        self.dx = dx

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_gx, pred_gy = spatial_gradients(pred, dx=self.dx)
        target_gx, target_gy = spatial_gradients(target, dx=self.dx)
        loss_x = F.mse_loss(pred_gx, target_gx)
        loss_y = F.mse_loss(pred_gy, target_gy)
        return loss_x + loss_y
    