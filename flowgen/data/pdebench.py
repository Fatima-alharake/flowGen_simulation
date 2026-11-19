from typing import Optional, Tuple, List
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .degradation import SpatialDegradation


class PDEBenchDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        scale_factor: int = 4,
        degradation_mode: str = "bicubic",
        channels: Optional[List[int]] = None,
        normalize: bool = True,
        stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.file_path = file_path
        self.split = split
        self.scale_factor = scale_factor
        self.degrader = SpatialDegradation(scale_factor=scale_factor, mode=degradation_mode)
        self.channels = channels
        self.normalize = normalize

        raw_data = self._load_data(file_path)
        if max_samples is not None:
            raw_data = raw_data[:max_samples]

        n_total = len(raw_data)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])

        if split == "train":
            subset = raw_data[:n_train]
        elif split == "val":
            subset = raw_data[n_train:n_train + n_val]
        elif split == "test":
            subset = raw_data[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.data = torch.from_numpy(subset).float()

        if self.normalize:
            if stats is not None:
                self.mean, self.std = stats
            else:
                self.mean = self.data.mean(dim=(0, 2, 3), keepdim=True)
                self.std = self.data.std(dim=(0, 2, 3), keepdim=True) + 1e-7
            self.data = (self.data - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

    def _load_data(self, file_path: str) -> np.ndarray:
        with h5py.File(file_path, "r") as f:
            candidates = ["data", "velocity", "u", "tensor", "fields"]
            key = next((k for k in candidates if k in f), None)
            if key is None:
                keys = list(f.keys())
                if not keys:
                    raise KeyError(f"No datasets found in {file_path}")
                key = keys[0]

            arr = np.array(f[key], dtype=np.float32)

        if arr.ndim == 5:
            n, t, h, w, c = arr.shape
            arr = arr.reshape(n * t, h, w, c)
            arr = np.transpose(arr, (0, 3, 1, 2))
        elif arr.ndim == 4:
            if arr.shape[1] > arr.shape[-1] and arr.shape[-1] in (1, 2, 3, 4):
                arr = np.transpose(arr, (0, 3, 1, 2))
        elif arr.ndim == 3:
            arr = np.expand_dims(arr, axis=1)

        if self.channels is not None:
            arr = arr[:, self.channels, :, :]

        return arr

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr = self.data[idx]
        lr = self.degrader(hr.unsqueeze(0)).squeeze(0)
        return lr, hr

    @property
    def normalization_stats(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.mean, self.std
