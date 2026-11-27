from typing import Dict, Tuple
import numpy as np
import torch
from ..losses.physical import compute_vorticity, compute_divergence


def radially_averaged_spectrum(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = field.shape
    fft2 = np.fft.fftshift(np.fft.fft2(field))
    power = np.abs(fft2) ** 2 / (h * w)

    y, x = np.indices((h, w))
    center = (h // 2, w // 2)
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(np.int64)

    max_k = min(center[0], center[1])
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin[:max_k] / np.maximum(nr[:max_k], 1)
    k_bins = np.arange(len(radial_profile))

    return k_bins, radial_profile


def compute_kinetic_energy_spectrum(velocity_field: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    u = velocity_field[:, 0].detach().cpu().numpy()
    v = velocity_field[:, 1].detach().cpu().numpy()

    batch_spectra = []
    k_bins = None
    for b in range(len(u)):
        _, e_u = radially_averaged_spectrum(u[b])
        k, e_v = radially_averaged_spectrum(v[b])
        k_bins = k
        batch_spectra.append(0.5 * (e_u + e_v))

    avg_spectrum = np.mean(batch_spectra, axis=0)
    return k_bins, avg_spectrum


def compute_physics_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    w_pred = compute_vorticity(pred)
    w_target = compute_vorticity(target)

    enstrophy_pred = 0.5 * torch.mean(w_pred**2).item()
    enstrophy_target = 0.5 * torch.mean(w_target**2).item()
    enstrophy_rel_err = abs(enstrophy_pred - enstrophy_target) / (enstrophy_target + 1e-7)

    w_pred_flat = w_pred.reshape(-1)
    w_target_flat = w_target.reshape(-1)
    vx = w_pred_flat - torch.mean(w_pred_flat)
    vy = w_target_flat - torch.mean(w_target_flat)
    vort_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2)) + 1e-7)

    div_pred = compute_divergence(pred)
    div_rms = torch.sqrt(torch.mean(div_pred**2)).item()

    k, e_pred = compute_kinetic_energy_spectrum(pred)
    _, e_target = compute_kinetic_energy_spectrum(target)
    spec_rel_l2 = np.linalg.norm(e_pred - e_target) / (np.linalg.norm(e_target) + 1e-7)

    return {
        "enstrophy_pred": float(enstrophy_pred),
        "enstrophy_target": float(enstrophy_target),
        "enstrophy_rel_err": float(enstrophy_rel_err),
        "vorticity_correlation": float(vort_corr.item()),
        "divergence_rms": float(div_rms),
        "spectrum_rel_l2": float(spec_rel_l2),
    }
