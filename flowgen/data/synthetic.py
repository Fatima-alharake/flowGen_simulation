import h5py
import numpy as np


def generate_divergence_free_field(grid_size: int = 128, spectrum_decay: float = 3.0, rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng()

    kx = np.fft.fftfreq(grid_size) * grid_size
    ky = np.fft.fftfreq(grid_size) * grid_size
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    k_sq = kx**2 + ky**2
    k_sq[0, 0] = 1.0

    amplitude = np.sqrt(k_sq ** (-spectrum_decay / 2.0))
    phase = rng.uniform(0, 2 * np.pi, size=(grid_size, grid_size))
    psi_hat = amplitude * (np.cos(phase) + 1j * np.sin(phase))
    psi_hat[0, 0] = 0.0

    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat

    u = np.real(np.fft.ifft2(u_hat))
    v = np.real(np.fft.ifft2(v_hat))

    u = (u - u.mean()) / (u.std() + 1e-7)
    v = (v - v.mean()) / (v.std() + 1e-7)
    return np.stack([u, v], axis=0).astype(np.float32)


def generate_synthetic_kolmogorov(
    output_path: str,
    num_samples: int = 500,
    grid_size: int = 128,
    spectrum_decay: float = 3.0,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    data = np.zeros((num_samples, 2, grid_size, grid_size), dtype=np.float32)

    for i in range(num_samples):
        data[i] = generate_divergence_free_field(
            grid_size=grid_size,
            spectrum_decay=spectrum_decay,
            rng=rng,
        )

    with h5py.File(output_path, "w") as f:
        f.create_dataset("data", data=data, compression="gzip")
        f.attrs["grid_size"] = grid_size
        f.attrs["num_samples"] = num_samples
        f.attrs["channels"] = ["u", "v"]

    return data
