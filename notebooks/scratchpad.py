#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

#%%
# Light grey -> dark red colormap
cmap = LinearSegmentedColormap.from_list(
    "lightgrey_to_darkred",
    ["#f0f0f0", "#b22222"],  # lighter grey, lighter red
    N=256,
)

fig, ax = plt.subplots(figsize=(6, 1.2), constrained_layout=True)
ax.set_axis_off()

norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
cb.set_label("light grey → dark red")

plt.show()

# %%

#%% Visualize MNIST samples (from dense_sparse_extractor.data)
from pathlib import Path

import torch

from dense_sparse_extractor.data import denormalize_mnist, make_mnist_datasets


def _to_display_image_mnist(x: torch.Tensor, *, normalized_like_mnist: bool) -> torch.Tensor:
    """
    Convert a MNIST-domain tensor (1,H,W) to displayable (H,W) in [0,1].
    """
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"Expected (1,H,W), got {tuple(x.shape)}")
    x01 = denormalize_mnist(x).clamp(0.0, 1.0) if normalized_like_mnist else x.clamp(0.0, 1.0)
    return x01[0]


repo_root = Path(__file__).resolve().parents[1]
data_dir = repo_root / "data"

mnist_normalized = True
mnist_train, _mnist_test = make_mnist_datasets(
    data_dir=data_dir,
    normalize=mnist_normalized,
    label_format="onehot",
    download=True,
)

g = torch.Generator().manual_seed(0)
n = 12
idxs = torch.randint(0, len(mnist_train), (n,), generator=g).tolist()

fig, axs = plt.subplots(3, 4, figsize=(10, 7))
axs = axs.flatten()
for k, idx in enumerate(idxs):
    x, y = mnist_train[idx]
    axs[k].imshow(_to_display_image_mnist(x, normalized_like_mnist=mnist_normalized), cmap="gray")
    # axs[k].set_title(f"idx={idx} cls={int(y.argmax().item())}")
    axs[k].axis("off")
plt.tight_layout()
plt.show()

#%% Visualize MNIST + lowfreq combined samples (reusing the sampled images; no CombinedDataset)
from dense_sparse_extractor.data import LowFreqTagConfig, LowFreqTagDataset

# Everything in THIS cell so it is self-contained and reuses the sampled images.
#
# Note: we combine in [0,1] space (not normalized space), otherwise visualization of sums
# is misleading (denormalizing a sum of normalized tensors).
mnist_train_01, _mnist_test_01 = make_mnist_datasets(
    data_dir=data_dir,
    normalize=False,
    label_format="onehot",
    download=True,
)
lowfreq_cfg_01 = LowFreqTagConfig(
    images_per_class=200,
    seed=0,
    normalize_like_mnist=False,
    cache_images=True,
    label_format="onehot",
    magnitude_num_images=2048,
    magnitude_seed=0,
    download=True,
    lowfreq_boost=0.8,
    augment=True,
    augment_jitter_max=2,
    augment_gaussian_std=0.02,
)
lowfreq_tags_01 = LowFreqTagDataset(lowfreq_cfg_01, data_dir=data_dir)


def _format_active(y: torch.Tensor) -> list[int]:
    return torch.nonzero(y > 0.5).flatten().tolist()


# Sample and store the underlying components first, then combine them directly.
g = torch.Generator().manual_seed(0)
n = 12
mnist_idxs = torch.randint(0, len(mnist_train_01), (n,), generator=g).tolist()
lowfreq_idxs = torch.randint(0, len(lowfreq_tags_01), (n,), generator=g).tolist()

mnist_xs: list[torch.Tensor] = []
mnist_ys: list[torch.Tensor] = []
lowfreq_xs: list[torch.Tensor] = []
lowfreq_ys: list[torch.Tensor] = []
combined_xs: list[torch.Tensor] = []
combined_ys: list[torch.Tensor] = []

for mi, li in zip(mnist_idxs, lowfreq_idxs, strict=True):
    x_m, y_m = mnist_train_01[mi]  # x_m: (1,28,28) in [0,1], y_m: onehot (10,)
    x_l, y_l = lowfreq_tags_01[li]  # x_l: (1,28,28) in [0,1], y_l: onehot (10,)
    x_c = (x_m + x_l).clamp(0.0, 1.0)
    y_c = (y_m + y_l).clamp(0.0, 1.0)
    mnist_xs.append(x_m)
    mnist_ys.append(y_m)
    lowfreq_xs.append(x_l)
    lowfreq_ys.append(y_l)
    combined_xs.append(x_c)
    combined_ys.append(y_c)

# Plot three separate 4x3 grids (3 rows x 4 cols): MNIST, lowfreq, combined.
# No titles/labels.
for imgs in (mnist_xs, lowfreq_xs, combined_xs):
    fig, axs = plt.subplots(3, 4, figsize=(10, 7))
    axs = axs.flatten()
    for k in range(n):
        axs[k].imshow(imgs[k][0], cmap="gray", vmin=0.0, vmax=1.0)
        axs[k].axis("off")
    plt.tight_layout()
    plt.show()

#%% Visualize MNIST + lowbit combined samples (reusing sampled images; no CombinedDataset)
from dense_sparse_extractor.data import LowBitTagConfig, LowBitTagDataset

# Combine in [0,1] space for correct visualization.
mnist_train_01, _mnist_test_01 = make_mnist_datasets(
    data_dir=data_dir,
    normalize=False,
    label_format="onehot",
    download=True,
)
lowbit_cfg_01 = LowBitTagConfig(
    images_per_class=200,
    seed=0,
    normalize_like_mnist=False,
    cache_images=True,
    label_format="onehot",
    p_on=0.03,
    on_min_brightness=1.0,
    augment=True,
    augment_jitter_max=0,
    augment_gaussian_std=0.0,
)
lowbit_tags_01 = LowBitTagDataset(lowbit_cfg_01)

g = torch.Generator().manual_seed(0)
n = 12
mnist_idxs = torch.randint(0, len(mnist_train_01), (n,), generator=g).tolist()
lowbit_idxs = torch.randint(0, len(lowbit_tags_01), (n,), generator=g).tolist()

mnist_xs = []
lowbit_xs = []
combined_xs = []
for mi, li in zip(mnist_idxs, lowbit_idxs, strict=True):
    x_m, _y_m = mnist_train_01[mi]  # (1,28,28) in [0,1]
    x_b, _y_b = lowbit_tags_01[li]  # (1,28,28) in [0,1]
    x_c = (x_m + x_b).clamp(0.0, 1.0)
    mnist_xs.append(x_m)
    lowbit_xs.append(x_b)
    combined_xs.append(x_c)

# Three separate 4x3 grids (3 rows x 4 cols): MNIST, lowbit, combined. No labels.
for imgs in (mnist_xs, lowbit_xs, combined_xs):
    fig, axs = plt.subplots(3, 4, figsize=(10, 7))
    axs = axs.flatten()
    for k in range(n):
        axs[k].imshow(imgs[k][0], cmap="gray", vmin=0.0, vmax=1.0)
        axs[k].axis("off")
    plt.tight_layout()
    plt.show()

#%% Visualize lowfreq tag samples (from dense_sparse_extractor.data)
from pathlib import Path

import torch

from dense_sparse_extractor.data import LowFreqTagConfig, LowFreqTagDataset, denormalize_mnist


def _to_display_image(x: torch.Tensor, *, normalized_like_mnist: bool = True) -> torch.Tensor:
    """
    Convert a MNIST-domain tensor (1,H,W) to displayable (H,W) in [0,1].
    """
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"Expected (1,H,W), got {tuple(x.shape)}")
    x01 = denormalize_mnist(x).clamp(0.0, 1.0) if normalized_like_mnist else x.clamp(0.0, 1.0)
    return x01[0]


# notebook file is at: repo_root/notebooks/scratchpad.py
repo_root = Path(__file__).resolve().parents[1]
data_dir = repo_root / "data"

lowfreq_cfg = LowFreqTagConfig(
    images_per_class=200,
    seed=0,
    normalize_like_mnist=True,
    cache_images=True,
    label_format="onehot",
    magnitude_num_images=2048,
    magnitude_seed=0,
    download=True,
    lowfreq_boost=0.8,
    augment=True,
    augment_jitter_max=2,
    augment_gaussian_std=0.02,
)
lowfreq_tags = LowFreqTagDataset(lowfreq_cfg, data_dir=data_dir)

g = torch.Generator().manual_seed(0)
n = 12
idxs = torch.randint(0, len(lowfreq_tags), (n,), generator=g).tolist()

fig, axs = plt.subplots(3, 4, figsize=(10, 7))
axs = axs.flatten()
for k, idx in enumerate(idxs):
    x, y = lowfreq_tags[idx]
    axs[k].imshow(_to_display_image(x, normalized_like_mnist=bool(lowfreq_cfg.normalize_like_mnist)), cmap="gray")
    #axs[k].set_title(f"idx={idx} cls={int(y.argmax().item())}")
    axs[k].axis("off")
plt.tight_layout()
plt.show()

#%% Visualize lowbit tag samples (from dense_sparse_extractor.data)
from dense_sparse_extractor.data import LowBitTagConfig, LowBitTagDataset

lowbit_cfg = LowBitTagConfig(
    images_per_class=200,
    seed=0,
    normalize_like_mnist=True,
    cache_images=True,
    label_format="onehot",
    p_on=0.03,
    on_min_brightness=1.0,
    augment=True,
    augment_jitter_max=0,
    augment_gaussian_std=0.0,
)
lowbit_tags = LowBitTagDataset(lowbit_cfg)

g = torch.Generator().manual_seed(0)
n = 12
idxs = torch.randint(0, len(lowbit_tags), (n,), generator=g).tolist()

fig, axs = plt.subplots(3, 4, figsize=(10, 7))
axs = axs.flatten()
for k, idx in enumerate(idxs):
    x, y = lowbit_tags[idx]
    axs[k].imshow(_to_display_image(x, normalized_like_mnist=bool(lowbit_cfg.normalize_like_mnist)), cmap="gray")
    #axs[k].set_title(f"idx={idx} cls={int(y.argmax().item())}")
    axs[k].axis("off")
plt.suptitle("LowBitTagDataset samples")
plt.tight_layout()
plt.show()


# %%
