from __future__ import annotations

#%% Imports
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from dense_sparse_extractor.data import (
    CombinedDataset,
    LowFreqTagConfig,
    LowFreqTagDataset,
    MNISTAugmentConfig,
    NoiseTagConfig,
    NoiseTagDataset,
    denormalize_mnist,
    make_mnist_datasets,
)


#%% Helpers
def to_display_image(x: torch.Tensor, *, normalized_like_mnist: bool = True) -> torch.Tensor:
    """
    Convert a MNIST-domain tensor (1,H,W) to displayable (H,W) in [0,1].
    """
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"Expected (1,H,W), got {tuple(x.shape)}")
    x01 = denormalize_mnist(x).clamp(0.0, 1.0) if normalized_like_mnist else x.clamp(0.0, 1.0)
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
    # Optional augmentation suite (train-only by default in loaders; this notebook
    # instantiates the dataset directly so augmentations apply here if enabled).
    augment=False,
    augment_shift_max=27,
    augment_gaussian_std=0.02,
)

lowfreq_cfg = LowFreqTagConfig(
    images_per_class=200,
    seed=0,
    normalize_like_mnist=True,
    cache_images=True,
    label_format="onehot",
    # Fixed magnitude spectrum estimated from MNIST (padded to 36x36)
    magnitude_num_images=2048,
    magnitude_seed=0,
    download=True,
    # Boost low frequencies in the fixed magnitude spectrum
    lowfreq_boost=0.8,
    # Augmentations
    augment=True,
    augment_jitter_max=2,
    augment_gaussian_std=0.02,
)


#%% Datasets
# MNIST (one-hot labels for combining)
mnist_train, _mnist_test = make_mnist_datasets(
    data_dir=data_dir, normalize=True, label_format="onehot", download=True
)

# MNIST with augmentations (used for augmentation visualization + combined demos)
mnist_aug_cfg = MNISTAugmentConfig(
    augment=True,
    brightness_clip_min=0.5,
    jitter_crop=2,
    gaussian_std=0.02,
    augment_apply_to_test=False,
)
mnist_train_aug, _mnist_test_aug = make_mnist_datasets(
    data_dir=data_dir,
    normalize=True,
    label_format="onehot",
    download=True,
    augment_cfg=mnist_aug_cfg,
)

# Noise tags: `images_per_class` random noise images per digit class
noise_ds = NoiseTagDataset(noise_cfg)

# Lowfreq tags: phase-randomized Fourier images (36->28 crop)
lowfreq_ds = LowFreqTagDataset(lowfreq_cfg, data_dir=data_dir)

# Combined dataset: pixelwise add images; multi-hot labels via OR
combined_ds = CombinedDataset([mnist_train, noise_ds], seed=0, num_classes=None)
combined_lowfreq_ds = CombinedDataset([mnist_train, lowfreq_ds], seed=0, num_classes=None)

# Combined datasets using MNIST augmentations
combined_ds_mnist_aug = CombinedDataset([mnist_train_aug, noise_ds], seed=0, num_classes=None)
combined_lowfreq_ds_mnist_aug = CombinedDataset([mnist_train_aug, lowfreq_ds], seed=0, num_classes=None)


#%% Sample a few items
idx = 1
x_mnist, y_mnist = mnist_train[idx]
x_noise, y_noise = noise_ds[idx]
x_comb, y_comb = combined_ds[idx]
x_lf, y_lf = lowfreq_ds[idx]
x_comb_lf, y_comb_lf = combined_lowfreq_ds[idx]
raw36 = lowfreq_ds.get_raw_36_01(idx)
raw28_center = raw36[:, 4:32, 4:32]


#%% Visualize (noise tags)
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
axs[0].imshow(to_display_image(x_mnist), cmap="gray", vmin=0.0, vmax=1.0)
axs[0].set_title(f"MNIST label: {y_mnist.argmax().item()}")
axs[0].axis("off")

axs[1].imshow(to_display_image(x_noise), cmap="gray", vmin=0.0, vmax=1.0)
axs[1].set_title(f"Noise tag label: {y_noise.argmax().item()}")
axs[1].axis("off")

axs[2].imshow(to_display_image(x_comb), cmap="gray", vmin=0.0, vmax=1.0)
active = torch.nonzero(y_comb > 0.5).flatten().tolist()
axs[2].set_title(f"Combined (MNIST+noise) multi-hot: {active}")
axs[2].axis("off")

plt.tight_layout()
plt.show()


#%% Visualize (MNIST augmentations: multiple views of ONE digit)
mnist_idx = idx
fig, axs = plt.subplots(2, 6, figsize=(12, 4))
axs = axs.flatten()

# Baseline (no MNIST aug)
x0, y0 = mnist_train[mnist_idx]
axs[0].imshow(to_display_image(x0), cmap="gray", vmin=0.0, vmax=1.0)
axs[0].set_title(f"orig (y={y0.argmax().item()})")
axs[0].axis("off")

# Multiple augmented samples of the same digit index
for k in range(1, 12):
    xk, yk = mnist_train_aug[mnist_idx]
    axs[k].imshow(to_display_image(xk), cmap="gray", vmin=0.0, vmax=1.0)
    axs[k].set_title(f"aug {k}")
    axs[k].axis("off")

plt.suptitle("MNIST digit: multiple augmentation samples (brightness clip + jitter crop + Gaussian noise)")
plt.tight_layout()
plt.show()

#%% Visualize (lowfreq tags: raw + cropped + combined)
fig, axs = plt.subplots(1, 4, figsize=(12, 3))
axs[0].imshow(to_display_image(raw36, normalized_like_mnist=False), cmap="gray", vmin=0.0, vmax=1.0)
axs[0].set_title("Lowfreq raw (36x36)")
axs[0].axis("off")

axs[1].imshow(to_display_image(raw28_center, normalized_like_mnist=False), cmap="gray", vmin=0.0, vmax=1.0)
axs[1].set_title("Center crop (28x28)")
axs[1].axis("off")

axs[2].imshow(to_display_image(x_lf), cmap="gray", vmin=0.0, vmax=1.0)
axs[2].set_title(f"Lowfreq tag (aug) label: {y_lf.argmax().item()}")
axs[2].axis("off")

axs[3].imshow(to_display_image(x_comb_lf), cmap="gray", vmin=0.0, vmax=1.0)
active_lf = torch.nonzero(y_comb_lf > 0.5).flatten().tolist()
axs[3].set_title(f"Combined (MNIST+lowfreq) multi-hot: {active_lf}")
axs[3].axis("off")

plt.tight_layout()
plt.show()


#%% Visualize (lowfreq: columns = base images, rows = augmentations)
def lowfreq_augmented_crop(
    *,
    raw36_01: torch.Tensor,
    jitter_max: int,
    gaussian_std: float,
    seed: int,
) -> torch.Tensor:
    """
    Create one augmented 28x28 view from a fixed raw 36x36 image.

    Augmentations:
    - crop jitter in [-jitter_max, +jitter_max] (within 36->28 margin)
    - additive Gaussian pixel noise with std `gaussian_std` in [0,1] space
    """
    if raw36_01.shape != (1, 36, 36):
        raise ValueError(f"Expected raw36_01 shape (1,36,36), got {tuple(raw36_01.shape)}")
    if jitter_max < 0:
        raise ValueError("jitter_max must be >= 0")
    margin = (36 - 28) // 2  # 4
    if jitter_max > margin:
        raise ValueError(f"jitter_max must be <= {margin}")

    g = torch.Generator()
    g.manual_seed(int(seed))

    dx = 0
    dy = 0
    if jitter_max > 0:
        dx = int(torch.randint(-jitter_max, jitter_max + 1, (1,), generator=g).item())
        dy = int(torch.randint(-jitter_max, jitter_max + 1, (1,), generator=g).item())

    top = margin + dy
    left = margin + dx
    x01 = raw36_01[:, top : top + 28, left : left + 28].clone()

    if float(gaussian_std) > 0.0:
        n = torch.randn(x01.shape, generator=g, device=x01.device, dtype=x01.dtype)
        x01 = (x01 + n * float(gaussian_std)).clamp_(0.0, 1.0)
    return x01


# Choose which base images to show as columns.
# Here we show one base image from each class (j=0), for the first N classes.
n_cols = 5
base_indices = [cls * int(lowfreq_cfg.images_per_class) for cls in range(n_cols)]
base_raw36 = [lowfreq_ds.get_raw_36_01(i) for i in base_indices]
base_classes = [int(i // int(lowfreq_cfg.images_per_class)) for i in base_indices]

# Choose how many augmentation rows (including a "no-aug center crop" first row).
n_rows = 6
jitter_max = int(lowfreq_cfg.augment_jitter_max)
gaussian_std = float(lowfreq_cfg.augment_gaussian_std)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.2 * n_rows))
if n_rows == 1 and n_cols == 1:
    axs = [[axs]]
elif n_rows == 1:
    axs = [axs]
elif n_cols == 1:
    axs = [[a] for a in axs]

for c in range(n_cols):
    for r in range(n_rows):
        ax = axs[r][c]
        if r == 0:
            # Baseline view: center crop, no added noise.
            x01 = base_raw36[c][:, 4:32, 4:32]
        else:
            # Different augmentation per row, per base image.
            # Seed mixes row + col to keep it deterministic but distinct.
            x01 = lowfreq_augmented_crop(
                raw36_01=base_raw36[c],
                jitter_max=jitter_max,
                gaussian_std=gaussian_std,
                seed=10_000 + 100 * r + c,
            )

        ax.imshow(to_display_image(x01, normalized_like_mnist=False), cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

        if r == 0:
            ax.set_title(f"class {base_classes[c]}\n(base idx {base_indices[c]})", fontsize=10)
        if c == 0:
            ax.set_ylabel("center\n(no aug)" if r == 0 else f"aug {r}", rotation=0, labelpad=28, va="center")

plt.suptitle("Lowfreq grid: columns = base images, rows = augmented views (crop-jitter + Gaussian noise)")
plt.tight_layout()
plt.show()


#%% Visualize (MNIST + lowfreq combined samples)
def format_active(y: torch.Tensor) -> list[int]:
    return torch.nonzero(y > 0.5).flatten().tolist()


n_comb = 8
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
axs = axs.flatten()
for i in range(n_comb):
    x, y = combined_lowfreq_ds_mnist_aug[idx + i]
    axs[i].imshow(to_display_image(x), cmap="gray", vmin=0.0, vmax=1.0)
    axs[i].set_title(f"active: {format_active(y)}")
    axs[i].axis("off")
plt.suptitle("Combined dataset samples (MNIST aug + lowfreq)")
plt.tight_layout()
plt.show()


# %%
