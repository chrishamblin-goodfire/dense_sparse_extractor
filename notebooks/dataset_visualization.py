from __future__ import annotations

#%% Imports
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from dense_sparse_extractor.data import (
    CombinedDataset,
    NoiseTagConfig,
    NoiseTagDataset,
    denormalize_mnist,
    make_mnist_datasets,
)


#%% Helpers
def to_display_image(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a MNIST-domain tensor (1,28,28) to displayable (28,28) in [0,1].
    """
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"Expected (1,H,W), got {tuple(x.shape)}")
    x01 = denormalize_mnist(x).clamp(0.0, 1.0)
    return x01[0]


#%% Config
# notebook file is at: repo_root/notebooks/dataset_visualization.py
repo_root = Path(__file__).resolve().parents[1]
data_dir = repo_root / "data"

noise_cfg = NoiseTagConfig(
    images_per_class=64,
    seed=0,
    normalize_like_mnist=True,
    distribution="uniform",
    cache_images=True,
    label_format="onehot",
)


#%% Datasets
# MNIST (one-hot labels for combining)
mnist_train, _mnist_test = make_mnist_datasets(
    data_dir=data_dir, normalize=True, label_format="onehot", download=True
)

# Noise tags: `images_per_class` random noise images per digit class
noise_ds = NoiseTagDataset(noise_cfg)

# Combined dataset: pixelwise add images; multi-hot labels via OR
combined_ds = CombinedDataset([mnist_train, noise_ds], seed=0, num_classes=None)


#%% Sample a few items
idx = 0
x_mnist, y_mnist = mnist_train[idx]
x_noise, y_noise = noise_ds[idx]
x_comb, y_comb = combined_ds[idx]


#%% Visualize
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
axs[0].imshow(to_display_image(x_mnist), cmap="gray")
axs[0].set_title(f"MNIST label: {y_mnist.argmax().item()}")
axs[0].axis("off")

axs[1].imshow(to_display_image(x_noise), cmap="gray")
axs[1].set_title(f"Noise tag label: {y_noise.argmax().item()}")
axs[1].axis("off")

axs[2].imshow(to_display_image(x_comb), cmap="gray")
active = torch.nonzero(y_comb > 0.5).flatten().tolist()
axs[2].set_title(f"Combined multi-hot: {active}")
axs[2].axis("off")

plt.tight_layout()
plt.show()


# %%
