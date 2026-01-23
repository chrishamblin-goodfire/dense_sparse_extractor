"""Dense/sparse feature extraction experiments package."""

from .data import (
    CombinedDataset,
    NoiseTagConfig,
    NoiseTagDataset,
    denormalize_mnist,
    make_mnist_datasets,
    make_mnist_loaders,
)
from .densae import DenSAE, DenSAEConfig

__all__ = [
    "CombinedDataset",
    "NoiseTagConfig",
    "NoiseTagDataset",
    "DenSAE",
    "DenSAEConfig",
    "denormalize_mnist",
    "make_mnist_datasets",
    "make_mnist_loaders",
]

