import os
from typing import Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
import torch
from ..losses.physical import compute_vorticity


def plot_reconstruction_sample(
    lr: torch.Tensor,
    hr: torch.Tensor,
    pred: torch.Tensor,
    save_path: str,
    uncertainty: Optional[torch.Tensor] = None,
    channel_idx: int = 0,
    channel_name: str = "Velocity u",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lr_np = lr[channel_idx].detach().cpu().numpy()
    hr_np = hr[channel_idx].detach().cpu().numpy()
    pred_np = pred[channel_idx].detach().cpu().numpy()
    err_np = np.abs(hr_np - pred_np)

    cols = 5 if uncertainty is not None else 4
    fig, axes = plt.subplots(1, cols, figsize=(3.8 * cols, 3.4), constrained_layout=True)

    vmin, vmax = min(hr_np.min(), pred_np.min()), max(hr_np.max(), pred_np.max())

    im0 = axes[0].imshow(lr_np, cmap="Spectral_r", origin="lower")
    axes[0].set_title(f"LR Input ({lr_np.shape[0]}x{lr_np.shape[1]})")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(hr_np, cmap="Spectral_r", vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title(f"Target HR ({hr_np.shape[0]}x{hr_np.shape[1]})")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pred_np, cmap="Spectral_r", vmin=vmin, vmax=vmax, origin="lower")
    axes[2].set_title(f"Reconstructed ({channel_name})")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(err_np, cmap="magma", origin="lower")
    axes[3].set_title("Absolute Error")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    if uncertainty is not None:
        unc_np = uncertainty[channel_idx].detach().cpu().numpy()
        im4 = axes[4].imshow(unc_np, cmap="viridis", origin="lower")
        axes[4].set_title("Predictive Std (Uncertainty)")
        fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_vorticity_comparison(
    hr: torch.Tensor,
    pred: torch.Tensor,
    save_path: str,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    w_hr = compute_vorticity(hr.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
    w_pred = compute_vorticity(pred.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
    w_err = np.abs(w_hr - w_pred)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)

    vlim = max(abs(w_hr.min()), abs(w_hr.max()))
    im0 = axes[0].imshow(w_hr, cmap="coolwarm", vmin=-vlim, vmax=vlim, origin="lower")
    axes[0].set_title("Target Vorticity $\\omega$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(w_pred, cmap="coolwarm", vmin=-vlim, vmax=vlim, origin="lower")
    axes[1].set_title("Predicted Vorticity $\\omega$")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(w_err, cmap="inferno", origin="lower")
    axes[2].set_title("Vorticity Residual $|\\omega - \\hat{\\omega}|$")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_energy_spectrum(
    k_bins: np.ndarray,
    spectra_dict: Dict[str, np.ndarray],
    save_path: str,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(6.2, 4.5))
    k = k_bins[1:]

    for label, spectrum in spectra_dict.items():
        spec = spectrum[1:]
        plt.loglog(k, spec, label=label, linewidth=1.8)

    ref_k = k[k > 2]
    if len(ref_k) > 0:
        c_ref = spectra_dict.get("Target HR", list(spectra_dict.values())[0])[3] * (ref_k[0] ** 3)
        plt.loglog(ref_k, c_ref * (ref_k**-3), "k--", alpha=0.5, label="$k^{-3}$ scaling")

    plt.xlabel("Wavenumber $k$")
    plt.ylabel("Kinetic Energy Spectrum $E(k)$")
    plt.title("Turbulent Energy Spectrum Preservation")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(frameon=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_uncertainty_calibration(
    pred_std: np.ndarray,
    abs_err: np.ndarray,
    save_path: str,
    num_bins: int = 15,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pred_std = pred_std.ravel()
    abs_err = abs_err.ravel()

    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(pred_std, quantiles)

    bin_centers = []
    bin_err_means = []

    for i in range(num_bins):
        mask = (pred_std >= bin_edges[i]) & (pred_std <= bin_edges[i + 1])
        if np.sum(mask) > 10:
            bin_centers.append(np.mean(pred_std[mask]))
            bin_err_means.append(np.mean(abs_err[mask]))

    plt.figure(figsize=(5.5, 4.5))
    plt.plot(bin_centers, bin_err_means, "o-", color="#1f77b4", label="Observed Error vs Std")

    if bin_centers:
        min_v = min(bin_centers[0], bin_err_means[0])
        max_v = max(bin_centers[-1], bin_err_means[-1])
        plt.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.6, label="Ideal Calibration")

    plt.xlabel("Predicted Uncertainty (Standard Deviation)")
    plt.ylabel("Mean Absolute Error")
    plt.title("Predictive Uncertainty Calibration")
    plt.grid(True, ls=":", alpha=0.5)
    plt.legend(frameon=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
