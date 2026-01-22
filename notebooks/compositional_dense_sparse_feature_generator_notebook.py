from __future__ import annotations

#%% Overview
# Part 1 (dense features): sample variable-size random crops from Tiny-ImageNet
# test images and visualize them.
#
# Data location (after running ./data/download_tiny_imagenet.sh):
#   dense_sparse_extractor/data/tiny-imagenet-200/test/images

#%% Imports
from dataclasses import dataclass
from pathlib import Path
import random

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.io import read_image
from torchvision.transforms import functional as TF


#%% Config
repo_root = Path(__file__).resolve().parents[1]
tiny_imagenet_test_images_dir = repo_root / "data" / "tiny-imagenet-200" / "test" / "images"

seed = 0
n_source_images = 10
crops_per_image = 1

# Crop sizing: choose random crop sizes in [min_frac, max_frac] of (H, W),
# clamped to [min_px, min(H,W)].
min_crop_frac = 0.20
max_crop_frac = 1.00
min_crop_px = 12


#%% Helpers
def seed_everything(s: int) -> None:
    random.seed(int(s))
    torch.manual_seed(int(s))


@dataclass(frozen=True)
class CropParams:
    top: int
    left: int
    height: int
    width: int


def sample_crop_params(*, height: int, width: int) -> CropParams:
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: (H,W)=({height},{width})")

    min_hw = min(height, width)
    h_min = max(min_crop_px, int(round(min_crop_frac * height)))
    w_min = max(min_crop_px, int(round(min_crop_frac * width)))
    h_max = max(h_min, int(round(max_crop_frac * height)))
    w_max = max(w_min, int(round(max_crop_frac * width)))

    # Make sure we don't exceed the image size.
    h_max = min(h_max, height)
    w_max = min(w_max, width)
    h_min = min(h_min, h_max)
    w_min = min(w_min, w_max)

    crop_h = random.randint(h_min, h_max)
    crop_w = random.randint(w_min, w_max)

    top = 0 if height == crop_h else random.randint(0, height - crop_h)
    left = 0 if width == crop_w else random.randint(0, width - crop_w)
    return CropParams(top=top, left=left, height=crop_h, width=crop_w)


def load_rgb_uint8(path: Path) -> torch.Tensor:
    """
    Returns uint8 tensor shaped (3,H,W) in RGB.
    """
    x = read_image(str(path))  # (C,H,W), uint8
    if x.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(x.shape)} from {path}")
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    if x.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {x.shape[0]} from {path}")
    return x


def to_imshow(x_uint8_chw: torch.Tensor) -> torch.Tensor:
    """
    Convert (3,H,W) uint8 to (H,W,3) float in [0,1] for matplotlib.
    """
    if x_uint8_chw.dtype != torch.uint8:
        raise ValueError(f"Expected uint8, got {x_uint8_chw.dtype}")
    if x_uint8_chw.ndim != 3 or x_uint8_chw.shape[0] != 3:
        raise ValueError(f"Expected (3,H,W), got {tuple(x_uint8_chw.shape)}")
    return (x_uint8_chw.permute(1, 2, 0).float() / 255.0).clamp(0.0, 1.0)


#%% Locate Tiny-ImageNet test images
seed_everything(seed)

if not tiny_imagenet_test_images_dir.exists():
    raise FileNotFoundError(
        "Tiny-ImageNet test images directory not found.\n"
        f"Expected: {tiny_imagenet_test_images_dir}\n"
        "Try running: ./data/download_tiny_imagenet.sh"
    )

image_paths = sorted(tiny_imagenet_test_images_dir.glob("*.JPEG"))
if len(image_paths) == 0:
    raise FileNotFoundError(f"No .JPEG images found in: {tiny_imagenet_test_images_dir}")

print(f"Found {len(image_paths)} test images at: {tiny_imagenet_test_images_dir}")


#%% Sample random images + variable-size random crops
picked_paths = random.sample(image_paths, k=min(n_source_images, len(image_paths)))

samples: list[dict[str, object]] = []
for p in picked_paths:
    img = load_rgb_uint8(p)
    _c, h, w = img.shape

    crops: list[tuple[torch.Tensor, CropParams]] = []
    for _ in range(int(crops_per_image)):
        params = sample_crop_params(height=h, width=w)
        crop = TF.crop(img, params.top, params.left, params.height, params.width)
        crops.append((crop, params))

    samples.append({"path": p, "image": img, "crops": crops})

print(f"Sampled {len(samples)} images and {len(samples) * int(crops_per_image)} crops total.")


#%% Visualize sampled crops (grid)
#
# Layout:
# - each row is one source image
# - first column: the full image
# - following columns: random crops (variable sizes)
n_rows = len(samples)
n_cols = 1 + int(crops_per_image)

fig_w = 3.0 * n_cols
fig_h = 2.8 * max(1, n_rows)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

if n_rows == 1 and n_cols == 1:
    axs = [[axs]]
elif n_rows == 1:
    axs = [axs]
elif n_cols == 1:
    axs = [[a] for a in axs]

for r, s in enumerate(samples):
    path = s["path"]
    img = s["image"]
    crops = s["crops"]

    ax0 = axs[r][0]
    ax0.imshow(to_imshow(img))
    ax0.set_title(f"source\n{Path(str(path)).name}", fontsize=9)
    ax0.axis("off")

    for c, (crop, params) in enumerate(crops, start=1):
        ax = axs[r][c]
        ax.imshow(to_imshow(crop))
        ax.set_title(f"{params.height}x{params.width}\n({params.top},{params.left})", fontsize=9)
        ax.axis("off")

plt.suptitle("Tiny-ImageNet test images: variable-size random crops (dense feature candidates)")
plt.tight_layout()
plt.show()


#%% Visualize crops only (montage)
# Useful to quickly scan the crop pool without the source images.
all_crops: list[tuple[torch.Tensor, CropParams, Path]] = []
for s in samples:
    for crop, params in s["crops"]:
        all_crops.append((crop, params, Path(str(s["path"]))))

n_show = min(len(all_crops), 48)
picked = random.sample(all_crops, k=n_show) if n_show > 0 else []

if n_show == 0:
    print("No crops to display.")
else:
    n_cols = 8
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    axs = axs.flatten() if hasattr(axs, "flatten") else [axs]

    for i, ax in enumerate(axs):
        ax.axis("off")
        if i >= n_show:
            continue
        crop, params, src = picked[i]
        ax.imshow(to_imshow(crop))
        ax.set_title(f"{params.height}x{params.width}", fontsize=9)

    plt.suptitle("Random sample of crops (dense feature candidates)")
    plt.tight_layout()
    plt.show()


#%% Visualize crop boxes overlaid on source images
# Shows where crops come from within the original image.
fig, axs = plt.subplots(len(samples), 1, figsize=(8.5, 3.5 * max(1, len(samples))))
if len(samples) == 1:
    axs = [axs]

for ax, s in zip(axs, samples):
    path = Path(str(s["path"]))
    img = s["image"]
    crops = s["crops"]

    ax.imshow(to_imshow(img))
    ax.set_title(f"source with crop boxes: {path.name}", fontsize=10)
    ax.axis("off")

    # Deterministic-ish distinct colors per crop
    for i, (_crop, params) in enumerate(crops):
        color = plt.cm.tab10(i % 10)
        rect = Rectangle(
            (params.left, params.top),
            params.width,
            params.height,
            fill=False,
            linewidth=2.0,
            edgecolor=color,
        )
        ax.add_patch(rect)
        ax.text(
            params.left,
            max(0, params.top - 2),
            f"{params.height}x{params.width}",
            color=color,
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.35, pad=1, edgecolor="none"),
        )

plt.tight_layout()
plt.show()


#%% Crop size / aspect ratio diagnostics
heights = []
widths = []
aspects = []
areas = []
for s in samples:
    for _crop, p in s["crops"]:
        heights.append(int(p.height))
        widths.append(int(p.width))
        aspects.append(float(p.width) / float(p.height))
        areas.append(int(p.width * p.height))

fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
axs[0].hist(heights, bins=20)
axs[0].set_title("crop heights")
axs[0].set_xlabel("px")

axs[1].hist(widths, bins=20)
axs[1].set_title("crop widths")
axs[1].set_xlabel("px")

axs[2].hist(aspects, bins=20)
axs[2].set_title("aspect ratio (W/H)")
axs[2].set_xlabel("ratio")

axs[3].hist(areas, bins=20)
axs[3].set_title("crop area")
axs[3].set_xlabel("px^2")

plt.tight_layout()
plt.show()


#%% (Optional) Save crops to disk for reuse
# Uncomment if you want to materialize the crop pool on disk.
#
# out_dir = repo_root / "data" / "dense_feature_crops_tiny_imagenet_test"
# out_dir.mkdir(parents=True, exist_ok=True)
# idx = 0
# for s in samples:
#     src_name = Path(str(s["path"])).stem
#     for crop, params in s["crops"]:
#         out_path = out_dir / f"{src_name}__{idx:06d}__{params.height}x{params.width}__t{params.top}_l{params.left}.png"
#         # torchvision doesn't have a direct PNG writer; easiest is PIL, but we avoid
#         # adding deps here. If you want saving, tell me and I'll wire in PIL properly.
#         idx += 1

