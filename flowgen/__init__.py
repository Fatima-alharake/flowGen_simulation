from .pdebench import PDEBenchDataset
from .degradation import SpatialDegradation
from .synthetic import generate_synthetic_kolmogorov

__all__ = ["PDEBenchDataset", "SpatialDegradation", "generate_synthetic_kolmogorov"]
