from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# MNIST normalization constants (train set)
MNIST_MEAN: float = 0.1307
MNIST_STD: float = 0.3081
MNIST_NUM_CLASSES: int = 10
MNIST_IMAGE_SHAPE = (1, 28, 28)

LabelFormat = Literal["int", "onehot"]
NoiseDistribution = Literal["uniform", "normal"]


def _stable_superellipse_radius(xu: np.ndarray, yv: np.ndarray, p_exp: float, eps: float = 1e-12) -> np.ndarray:
    """
    Stable computation of r = (xu^p + yv^p)^(1/p) for p>0, using log-sum-exp.

    Adapted from `notebooks/loop_generator.py`.
    """
    p_exp = float(p_exp)
    if not (p_exp > 0):
        raise ValueError(f"p_exp must be > 0, got {p_exp}")

    a = p_exp * np.log(xu + eps)
    b = p_exp * np.log(yv + eps)
    m = np.maximum(a, b)
    log_sum = m + np.log(np.exp(a - m) + np.exp(b - m))
    return np.exp(log_sum / p_exp)


def generate_loop_image(
    *,
    size: float,
    tx: float,
    ty: float,
    aspect: float,
    rotation: float,
    thickness: float,
    brightness: float,
    p_shape: float,
    image_size: int = 28,
    antialias: bool = True,
) -> np.ndarray:
    """
    Generate a single (image_size x image_size) grayscale image with a loop (ring).

    Output is float32 in [0,1].

    Adapted from `notebooks/loop_generator.py`.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if size <= 0:
        raise ValueError("size must be > 0")
    if thickness <= 0:
        raise ValueError("thickness must be > 0")
    if aspect < 1:
        raise ValueError("aspect must be >= 1 (major/minor ratio)")

    brightness = float(np.clip(brightness, 0.5, 1.0))

    # Semi-axes in pixels
    b = float(size)  # minor
    a = float(size * aspect)  # major

    # Coordinate grid in pixel units, centered at image center.
    c = (image_size - 1) / 2.0
    xs = np.arange(image_size, dtype=np.float32)
    ys = np.arange(image_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dx = (xx - c) - float(tx)
    dy = (yy - c) - float(ty)

    # Rotate coordinates by -rotation to rotate the shape by +rotation.
    ct = float(np.cos(rotation))
    st = float(np.sin(rotation))
    u = ct * dx + st * dy
    v = -st * dx + ct * dy

    xu = np.abs(u) / a
    yv = np.abs(v) / b

    p_exp = 2.0 * float(np.exp(p_shape))
    p_exp = float(np.clip(p_exp, 1e-3, 1e6))
    r = _stable_superellipse_radius(xu, yv, p_exp=p_exp)  # boundary at r=1

    # Convert thickness (pixels) into a band thickness in r-units (approx).
    mean_radius = 0.5 * (a + b)
    half_band = 0.5 * float(thickness) / mean_radius
    # Anti-alias band edges across ~1 pixel in r-units.
    edge = (0.5 / mean_radius) if antialias else 1e-9

    dist = np.abs(r - 1.0)
    alpha = np.clip((half_band - dist) / edge + 0.5, 0.0, 1.0)
    img = (brightness * alpha).astype(np.float32)
    return img


@dataclass(frozen=True)
class LoopConfig:
    """
    Synthetic 28x28 loop image dataset.

    Sampling ranges match `notebooks/loop_generator.py`.
    """

    train_length: int = 60_000
    test_length: int = 10_000
    seed: int = 0

    image_size: int = 28
    antialias: bool = True

    # Output domain control.
    normalize_like_mnist: bool = False
    label_format: LabelFormat = "onehot"

    # Parameter ranges (copied from the demo in loop_generator.py)
    size_min: float = 3.0
    size_max: float = 6.0
    tx_min: float = -10.0
    tx_max: float = 10.0
    ty_min: float = -10.0
    ty_max: float = 10.0
    aspect_min: float = 1.0
    aspect_max: float = 3.0
    rotation_min: float = 0.0
    rotation_max: float = 2 * math.pi
    thickness_min: float = 1.0
    thickness_max: float = 4.0
    brightness_min: float = 0.5
    brightness_max: float = 1.0
    p_shape_min: float = 0.0
    p_shape_max: float = 2.0


class LoopDataset(Dataset):
    """
    On-the-fly synthetic loop images.

    Returns:
      x: (1,28,28) tensor, either in [0,1] or MNIST-normalized space
      y:
        - int (always 0), if cfg.label_format == "int"
        - float vector shape (MNIST_NUM_CLASSES,), all zeros, if cfg.label_format == "onehot"

    The all-zeros onehot is intentional: loops have no digit tag, but this allows
    clean composition inside `CombinedDataset` (multi-hot OR).
    """

    def __init__(self, cfg: LoopConfig, *, split: Literal["train", "test"]):
        self.cfg = cfg
        self.split = str(split)
        if self.split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.epoch: int = 0

    def __len__(self) -> int:
        return int(self.cfg.train_length if self.split == "train" else self.cfg.test_length)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, idx: int) -> np.random.Generator:
        split_offset = 0 if self.split == "train" else 1_000_000_007
        seed = int(self.cfg.seed) + split_offset + int(idx) * 10_007 + int(self.epoch) * 2_000_033
        return np.random.default_rng(seed)

    def _postprocess(self, x01: torch.Tensor) -> torch.Tensor:
        if bool(self.cfg.normalize_like_mnist):
            return (x01 - MNIST_MEAN) / MNIST_STD
        return x01

    def _format_label(self):
        if self.cfg.label_format == "int":
            return 0
        return torch.zeros((MNIST_NUM_CLASSES,), dtype=torch.float32)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        rng = self._rng(int(idx))

        size = float(rng.uniform(self.cfg.size_min, self.cfg.size_max))
        tx = float(rng.uniform(self.cfg.tx_min, self.cfg.tx_max))
        ty = float(rng.uniform(self.cfg.ty_min, self.cfg.ty_max))
        aspect = float(rng.uniform(self.cfg.aspect_min, self.cfg.aspect_max))
        rotation = float(rng.uniform(self.cfg.rotation_min, self.cfg.rotation_max))
        thickness = float(rng.uniform(self.cfg.thickness_min, self.cfg.thickness_max))
        brightness = float(rng.uniform(self.cfg.brightness_min, self.cfg.brightness_max))
        p_shape = float(rng.uniform(self.cfg.p_shape_min, self.cfg.p_shape_max))

        img = generate_loop_image(
            size=size,
            tx=tx,
            ty=ty,
            aspect=aspect,
            rotation=rotation,
            thickness=thickness,
            brightness=brightness,
            p_shape=p_shape,
            image_size=int(self.cfg.image_size),
            antialias=bool(self.cfg.antialias),
        )  # (28,28) float32 in [0,1]

        x01 = torch.from_numpy(img).unsqueeze(0).to(torch.float32)  # (1,28,28)
        x = self._postprocess(x01)
        y = self._format_label()
        return x, y


def make_loop_datasets(*, cfg: LoopConfig) -> tuple[Dataset, Dataset]:
    train_ds = LoopDataset(cfg, split="train")
    # Ensure eval uses same base sampling seed by default.
    test_cfg = replace(cfg, seed=int(cfg.seed))
    test_ds = LoopDataset(test_cfg, split="test")
    return train_ds, test_ds


def make_loop_loaders(
    *,
    cfg: LoopConfig,
    batch_size: int,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds, test_ds = make_loop_datasets(cfg=cfg)

    effective_pin_memory = bool(pin_memory)
    if device is not None:
        effective_pin_memory = effective_pin_memory and device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle_train),
        num_workers=int(num_workers),
        pin_memory=effective_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=effective_pin_memory,
    )
    return train_loader, test_loader


@dataclass(frozen=True)
class MNISTAugmentConfig:
    """
    Optional MNIST augmentations applied in [0,1] pixel space (before normalization).

    - brightness clipping: keeps pixels <= clip_min unchanged, but compresses pixels
      above clip_min so that 1.0 maps to a randomly sampled max brightness in
      [clip_min, 1.0].
    - jitter crop: zero-pad by jitter pixels then random crop back to 28x28
    - gaussian noise: additive N(0, gaussian_std) in [0,1] space
    """

    augment: bool = True
    # Brightness clipping parameter. Set to 0.0 to disable.
    # Typical value: 0.5
    brightness_clip_min: float = 0.5
    # Jitter crop in pixels. 0 disables.
    jitter_crop: int = 0
    # Additive Gaussian noise std in [0,1] space. 0 disables.
    gaussian_std: float = 0.03
    # Apply augmentations to test split (default: train-only).
    augment_apply_to_test: bool = False


class _MNISTRandomBrightnessCeiling:
    def __init__(self, *, clip_min: float):
        self.clip_min = float(clip_min)
        if self.clip_min < 0.0 or self.clip_min > 1.0:
            raise ValueError("brightness_clip_min must be in [0,1]")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected (1,28,28) in [0,1]
        c = float(self.clip_min)
        if c <= 0.0:
            return x
        # Sample a per-image max brightness in [c, 1]
        bmax = float(torch.empty((), dtype=torch.float32).uniform_(c, 1.0).item())
        if bmax >= 1.0:
            return x
        # Keep values <= c fixed; compress values above c so that 1 -> bmax.
        above = x > c
        if not bool(above.any()):
            return x
        scale = (bmax - c) / max(1e-8, 1.0 - c)
        out = x.clone()
        out[above] = c + (out[above] - c) * scale
        return out


class _MNISTAddGaussianNoise:
    def __init__(self, *, std: float):
        self.std = float(std)
        if self.std < 0.0:
            raise ValueError("gaussian_std must be >= 0")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std == 0.0:
            return x
        n = torch.randn(x.shape, dtype=x.dtype, device=x.device) * self.std
        return (x + n).clamp_(0.0, 1.0)


def mnist_transform(*, normalize: bool = True, augment_cfg: MNISTAugmentConfig | None = None) -> transforms.Compose:
    tfms: list[transforms.Transform] = [transforms.ToTensor()]

    if augment_cfg is not None and bool(augment_cfg.augment):
        j = int(augment_cfg.jitter_crop)
        if j < 0:
            raise ValueError("jitter_crop must be >= 0")
        if j > 0:
            tfms.append(transforms.Pad(padding=j, fill=0))
            tfms.append(transforms.RandomCrop(size=28))

        c = float(augment_cfg.brightness_clip_min)
        if c > 0.0:
            tfms.append(_MNISTRandomBrightnessCeiling(clip_min=c))

        std = float(augment_cfg.gaussian_std)
        if std > 0.0:
            tfms.append(_MNISTAddGaussianNoise(std=std))

    if normalize:
        tfms.append(transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    return transforms.Compose(tfms)


class LabelFormatWrapper(Dataset):
    """Wrap a (x, y_int) dataset and optionally convert y to one-hot."""

    def __init__(self, base: Dataset, *, num_classes: int, label_format: LabelFormat):
        self.base = base
        self.num_classes = int(num_classes)
        self.label_format: LabelFormat = label_format

    def __len__(self) -> int:  # pragma: no cover
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        if self.label_format == "int":
            # torchvision MNIST returns Python int labels
            return x, int(y)

        y_t = torch.as_tensor(int(y), dtype=torch.long)
        y_oh = F.one_hot(y_t, num_classes=self.num_classes).to(torch.float32)
        return x, y_oh


def make_mnist_datasets(
    *,
    data_dir: str | Path,
    normalize: bool = True,
    label_format: LabelFormat = "int",
    download: bool = True,
    augment_cfg: MNISTAugmentConfig | None = None,
) -> tuple[Dataset, Dataset]:
    root = str(Path(data_dir))
    train_tfm = mnist_transform(normalize=bool(normalize), augment_cfg=augment_cfg)
    # By default, test split gets no augmentation.
    test_aug = None
    if augment_cfg is not None and bool(augment_cfg.augment) and bool(augment_cfg.augment_apply_to_test):
        test_aug = augment_cfg
    test_tfm = mnist_transform(normalize=bool(normalize), augment_cfg=test_aug)
    train_base = datasets.MNIST(root=root, train=True, download=bool(download), transform=train_tfm)
    test_base = datasets.MNIST(root=root, train=False, download=bool(download), transform=test_tfm)

    if label_format == "int":
        return train_base, test_base

    train_ds = LabelFormatWrapper(train_base, num_classes=MNIST_NUM_CLASSES, label_format=label_format)
    test_ds = LabelFormatWrapper(test_base, num_classes=MNIST_NUM_CLASSES, label_format=label_format)
    return train_ds, test_ds


def make_mnist_loaders(
    *,
    data_dir: str | Path,
    batch_size: int,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    normalize: bool = True,
    label_format: LabelFormat = "int",
    download: bool = True,
    augment_cfg: MNISTAugmentConfig | None = None,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds, test_ds = make_mnist_datasets(
        data_dir=data_dir,
        normalize=normalize,
        label_format=label_format,
        download=download,
        augment_cfg=augment_cfg,
    )

    effective_pin_memory = bool(pin_memory)
    if device is not None:
        effective_pin_memory = effective_pin_memory and device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle_train),
        num_workers=int(num_workers),
        pin_memory=effective_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=effective_pin_memory,
    )
    return train_loader, test_loader


def make_noise_tag_datasets(*, cfg: NoiseTagConfig) -> tuple[Dataset, Dataset]:
    """
    Create train/test synthetic noise-tag datasets.

    Train/test synthetic noise-tag datasets.

    By design, we evaluate on the *same underlying noise images* as training
    (same cfg.seed). This avoids any "unseen noise image" generalization
    requirement for the noise-tag setting.

    Both splits always share the same tag vocabulary (cfg.classes, or digits 0..9).

    By default, augmentations (if enabled in cfg) are applied to train only.
    """
    train_ds = NoiseTagDataset(cfg)
    # Keep the same base noise images for evaluation.
    test_cfg = replace(cfg, seed=int(cfg.seed))
    if bool(cfg.augment) and not bool(cfg.augment_apply_to_test):
        test_cfg = replace(
            test_cfg,
            augment=False,
            augment_shift_max=0,
            augment_gaussian_std=0.0,
        )
    test_ds = NoiseTagDataset(test_cfg)
    return train_ds, test_ds


def make_noise_tag_loaders(
    *,
    cfg: NoiseTagConfig,
    batch_size: int,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds, test_ds = make_noise_tag_datasets(cfg=cfg)

    effective_pin_memory = bool(pin_memory)
    if device is not None:
        effective_pin_memory = effective_pin_memory and device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle_train),
        num_workers=int(num_workers),
        pin_memory=effective_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=effective_pin_memory,
    )
    return train_loader, test_loader


@dataclass(frozen=True)
class LowFreqTagConfig:
    """
    Synthetic "tag" dataset generated in the Fourier domain.

    Base generation:
    - compute a fixed magnitude spectrum from "natural" images (by default: mean
      magnitude spectrum of MNIST train images padded to 36x36)
    - sample phase by taking the phase of spatial white noise (ensures Hermitian
      structure for real-valued inverse FFT)
    - inverse FFT to spatial domain, then map to [0,1]

    Output:
    - crop from 36x36 to 28x28 (optionally with small jitter)
    - optionally add Gaussian pixel noise (in [0,1] space)
    - optionally normalize like MNIST
    """

    images_per_class: int = 200
    seed: int = 0
    # If True (default), train/test will share the same underlying base images (same seed),
    # and "generalization" is evaluated mainly over augmentation/noise.
    # If False, test uses a different seed so base images differ across splits.
    test_split_same_images: bool = True
    normalize_like_mnist: bool = True
    cache_images: bool = True
    label_format: LabelFormat = "onehot"
    classes: tuple[int, ...] | None = None

    # Magnitude spectrum source.
    magnitude_num_images: int = 2048
    magnitude_seed: int = 0
    download: bool = True
    # Scalar low-frequency emphasis applied to the fixed magnitude spectrum.
    # Implemented as: mag *= (r_norm + eps)^(-lowfreq_boost), with a global rescale to
    # keep mean magnitude constant. 0.0 disables (default behavior).
    lowfreq_boost: float = 0.8

    # Spatial sizes (fixed for now; required by request).
    base_size: int = 36
    crop_size: int = 28

    # Optional augmentations (applied in [0,1] space before normalization).
    augment: bool = True
    augment_jitter_max: int = 2
    augment_gaussian_std: float = 0.0
    augment_apply_to_test: bool = False


_LOWFREQ_MAG_CACHE: dict[tuple[str, int, int, bool, int, float], torch.Tensor] = {}


def _apply_lowfreq_boost_to_mag(mag: torch.Tensor, *, boost: float) -> torch.Tensor:
    """
    Apply a radial low-frequency boost to an rfft2 magnitude spectrum.

    mag: shape (H, W//2+1)
    boost: scalar exponent. 0.0 -> unchanged.
    """
    b = float(boost)
    if b == 0.0:
        return mag

    if mag.ndim != 2:
        raise ValueError(f"Expected 2D magnitude spectrum, got shape={tuple(mag.shape)}")
    H = int(mag.shape[0])
    W2 = int(mag.shape[1])
    # For rfft2: second dimension is W//2+1, so infer W.
    W = (W2 - 1) * 2

    fy = torch.fft.fftfreq(H, d=1.0, device=mag.device)  # [-0.5, 0.5)
    fx = torch.fft.rfftfreq(W, d=1.0, device=mag.device)  # [0, 0.5]
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    r_max = float(r.max().item()) if r.numel() else 1.0
    r_norm = r / max(1e-12, r_max)

    # Avoid singularity at DC; keep eps tied to resolution.
    eps = 1.0 / max(1.0, float(H))
    w = (r_norm + eps).pow(-b)

    # Preserve overall scale (so downstream spatial mapping stays comparable).
    orig_mean = mag.mean()
    out = mag * w.to(dtype=mag.dtype)
    out_mean = out.mean()
    if float(out_mean.item()) != 0.0:
        out = out * (orig_mean / out_mean)
    return out


def _mnist_mean_magnitude_rfft2(
    *,
    data_dir: str | Path,
    image_size: int,
    num_images: int,
    seed: int,
    download: bool,
) -> torch.Tensor:
    """
    Compute mean |rfft2| magnitude over MNIST train images padded to `image_size`.

    Returns a float tensor of shape (image_size, image_size//2 + 1).
    """
    root = str(Path(data_dir))
    ds = datasets.MNIST(root=root, train=True, download=bool(download), transform=transforms.ToTensor())

    n = int(num_images)
    if n <= 0:
        raise ValueError("magnitude_num_images must be > 0")
    n = min(n, len(ds))

    pad_total = int(image_size) - 28
    if pad_total < 0:
        raise ValueError("image_size must be >= 28 for MNIST-based magnitude")
    if pad_total % 2 != 0:
        raise ValueError("image_size - 28 must be even (for symmetric padding)")
    pad_each = pad_total // 2

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(len(ds), generator=g)[:n]

    mag_sum = torch.zeros((int(image_size), int(image_size) // 2 + 1), dtype=torch.float32)
    bs = 256
    for start in range(0, n, bs):
        idxs = perm[start : start + bs].tolist()
        batch: list[torch.Tensor] = []
        for i in idxs:
            x01, _y = ds[int(i)]  # (1,28,28) in [0,1]
            if pad_each > 0:
                x01 = F.pad(x01, (pad_each, pad_each, pad_each, pad_each), mode="constant", value=0.0)
            batch.append(x01)
        x = torch.stack(batch, dim=0)  # (B,1,H,W)
        X = torch.fft.rfft2(x[:, 0], dim=(-2, -1))  # (B,H,W//2+1) complex
        mag_sum += X.abs().float().sum(dim=0)

    mag = mag_sum / float(n)
    return mag.clamp_min_(1e-8)


class LowFreqTagDataset(Dataset):
    """
    A synthetic "tag" dataset generated by randomizing Fourier phase while keeping a
    fixed (data-derived) magnitude spectrum.

    Output images are (1,28,28) in either MNIST-normalized space or [0,1] space.
    """

    def __init__(self, cfg: LowFreqTagConfig, *, data_dir: str | Path):
        self.cfg = cfg
        self.epoch: int = 0
        self.images_per_class = int(cfg.images_per_class)
        if self.images_per_class <= 0:
            raise ValueError("images_per_class must be > 0")

        base_size = int(cfg.base_size)
        crop_size = int(cfg.crop_size)
        if base_size != 36 or crop_size != 28:
            raise ValueError("LowFreqTagDataset currently requires base_size=36 and crop_size=28")

        # Which digit-tags exist in this dataset.
        if cfg.classes is None:
            classes = list(range(MNIST_NUM_CLASSES))
        else:
            classes = [int(c) for c in cfg.classes]
            if not classes:
                raise ValueError("classes must be None or a non-empty tuple of ints")
            if any((c < 0 or c >= MNIST_NUM_CLASSES) for c in classes):
                raise ValueError(f"classes must be in [0, {MNIST_NUM_CLASSES - 1}]")
            if len(set(classes)) != len(classes):
                raise ValueError("classes must not contain duplicates")
        self.classes: tuple[int, ...] = tuple(classes)

        cache_key = (
            str(Path(data_dir).resolve()),
            int(cfg.base_size),
            int(cfg.magnitude_num_images),
            bool(cfg.download),
            int(cfg.magnitude_seed),
            float(cfg.lowfreq_boost),
        )
        if cache_key in _LOWFREQ_MAG_CACHE:
            mag = _LOWFREQ_MAG_CACHE[cache_key]
        else:
            mag = _mnist_mean_magnitude_rfft2(
                data_dir=data_dir,
                image_size=int(cfg.base_size),
                num_images=int(cfg.magnitude_num_images),
                seed=int(cfg.magnitude_seed),
                download=bool(cfg.download),
            )
            mag = _apply_lowfreq_boost_to_mag(mag, boost=float(cfg.lowfreq_boost))
            _LOWFREQ_MAG_CACHE[cache_key] = mag
        self._mag = mag  # (36, 19)

        # Cache base 36x36 images in [0,1] for deterministic crop/noise augmentation.
        self._cache_36_01: torch.Tensor | None = None
        if bool(cfg.cache_images):
            self._cache_36_01 = self._generate_all_36_01()

    def __len__(self) -> int:
        return len(self.classes) * self.images_per_class

    def _postprocess(self, x01: torch.Tensor) -> torch.Tensor:
        if bool(self.cfg.normalize_like_mnist):
            return (x01 - MNIST_MEAN) / MNIST_STD
        return x01

    def _format_label(self, cls: int):
        if self.cfg.label_format == "int":
            return int(cls)
        y = F.one_hot(torch.as_tensor(int(cls), dtype=torch.long), num_classes=MNIST_NUM_CLASSES)
        return y.to(torch.float32)

    def _generate_one_36_01(self, cls: int, j: int) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(int(self.cfg.seed) + cls * 1_000_003 + j * 10_007)

        # Sample phase via spatial noise (valid real irfft2).
        u = torch.randn((36, 36), generator=g, dtype=torch.float32)
        U = torch.fft.rfft2(u, dim=(-2, -1))
        phase = U / (U.abs() + 1e-8)

        S = self._mag.to(phase.dtype) * phase
        x = torch.fft.irfft2(S, s=(36, 36), dim=(-2, -1))

        # Map to MNIST-ish intensity range, clamp to [0,1].
        x = x - x.mean()
        x = x / (x.std(unbiased=False) + 1e-6)
        x01 = (x * 0.2 + 0.5).clamp_(0.0, 1.0)
        return x01.unsqueeze(0)  # (1,36,36)

    def _generate_all_36_01(self) -> torch.Tensor:
        imgs = torch.empty((len(self.classes), self.images_per_class, 1, 36, 36), dtype=torch.float32)
        for i, cls in enumerate(self.classes):
            for j in range(self.images_per_class):
                imgs[i, j] = self._generate_one_36_01(int(cls), j)
        return imgs

    def _maybe_augment_and_crop_01(self, x36_01: torch.Tensor, *, cls: int, j: int) -> torch.Tensor:
        # Epoch-dependent per-sample augmentation.
        g = torch.Generator()
        g.manual_seed(int(self.cfg.seed) + cls * 1_000_003 + j * 10_007 + 424_243 + int(self.epoch) * 2_000_033)

        margin = (36 - 28) // 2  # 4
        dx = 0
        dy = 0
        if bool(self.cfg.augment):
            m = int(self.cfg.augment_jitter_max)
            if m < 0:
                raise ValueError("augment_jitter_max must be >= 0")
            if m > margin:
                raise ValueError(f"augment_jitter_max must be <= {margin}")
            if m > 0:
                dx = int(torch.randint(-m, m + 1, (1,), generator=g).item())
                dy = int(torch.randint(-m, m + 1, (1,), generator=g).item())

        top = margin + dy
        left = margin + dx
        x01 = x36_01[:, top : top + 28, left : left + 28]

        std = float(self.cfg.augment_gaussian_std) if bool(self.cfg.augment) else 0.0
        if std > 0.0:
            n = torch.randn(x01.shape, generator=g, dtype=x01.dtype) * std
            x01 = (x01 + n).clamp_(0.0, 1.0)

        return x01

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def get_raw_36_01(self, idx: int) -> torch.Tensor:
        """
        Debug helper: returns (1,36,36) in [0,1] (no crop, no augmentation).
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        cls_i = int(idx // self.images_per_class)
        cls = int(self.classes[cls_i])
        j = int(idx % self.images_per_class)
        if self._cache_36_01 is not None:
            return self._cache_36_01[cls_i, j].clone()
        return self._generate_one_36_01(cls, j)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        cls_i = int(idx // self.images_per_class)
        cls = int(self.classes[cls_i])
        j = int(idx % self.images_per_class)

        if self._cache_36_01 is not None:
            x36_01 = self._cache_36_01[cls_i, j].clone()
        else:
            x36_01 = self._generate_one_36_01(cls, j)

        x01 = self._maybe_augment_and_crop_01(x36_01, cls=cls, j=j)
        x = self._postprocess(x01)
        y = self._format_label(cls)
        return x, y


def make_lowfreq_tag_datasets(*, data_dir: str | Path, cfg: LowFreqTagConfig) -> tuple[Dataset, Dataset]:
    train_ds = LowFreqTagDataset(cfg, data_dir=data_dir)
    test_seed = int(cfg.seed) if bool(cfg.test_split_same_images) else int(cfg.seed) + 1
    test_cfg = replace(cfg, seed=test_seed)
    if bool(cfg.augment) and not bool(cfg.augment_apply_to_test):
        test_cfg = replace(test_cfg, augment=False, augment_jitter_max=0, augment_gaussian_std=0.0)
    test_ds = LowFreqTagDataset(test_cfg, data_dir=data_dir)
    return train_ds, test_ds


def make_lowfreq_tag_loaders(
    *,
    data_dir: str | Path,
    cfg: LowFreqTagConfig,
    batch_size: int,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds, test_ds = make_lowfreq_tag_datasets(data_dir=data_dir, cfg=cfg)

    effective_pin_memory = bool(pin_memory)
    if device is not None:
        effective_pin_memory = effective_pin_memory and device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle_train),
        num_workers=int(num_workers),
        pin_memory=effective_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=effective_pin_memory,
    )
    return train_loader, test_loader


@dataclass(frozen=True)
class LowBitTagConfig:
    """
    Synthetic "tag" dataset generated by sparse Bernoulli "bit" images.

    For each class c in {0..9}, generate `images_per_class` random binary images
    where each pixel is 1 with probability `p_on` and 0 otherwise.

    This mirrors the overall API/behavior of `LowFreqTagConfig`:
    - deterministic per-(class, index) generation via seed
    - optional caching
    - optional epoch-dependent augmentations
    - optional "test split uses same base images" behavior
    """

    images_per_class: int = 200
    seed: int = 0
    # If True (default), train/test will share the same underlying base images (same seed),
    # and "generalization" is evaluated mainly over augmentation/noise.
    # If False, test uses a different seed so base images differ across splits.
    test_split_same_images: bool = True
    normalize_like_mnist: bool = True
    cache_images: bool = True
    label_format: LabelFormat = "onehot"
    classes: tuple[int, ...] | None = None

    # Bernoulli sparsity for pixel "on" probability.
    p_on: float = 0.01  # ~1/100

    # Optional augmentations (applied in [0,1] space before normalization).
    augment: bool = True
    # Jitter implemented as cyclic shift (wrap-around) in pixels. 0 disables.
    augment_jitter_max: int = 2
    augment_gaussian_std: float = 0.0
    augment_apply_to_test: bool = False


class LowBitTagDataset(Dataset):
    """
    A synthetic "tag" dataset in the MNIST image domain using sparse Bernoulli bits.

    Output images are (1,28,28) in either MNIST-normalized space or [0,1] space.
    """

    def __init__(self, cfg: LowBitTagConfig):
        self.cfg = cfg
        self.epoch: int = 0

        self.images_per_class = int(cfg.images_per_class)
        if self.images_per_class <= 0:
            raise ValueError("images_per_class must be > 0")

        p = float(cfg.p_on)
        if p < 0.0 or p > 1.0:
            raise ValueError("p_on must be in [0,1]")
        self.p_on = p

        # Which digit-tags exist in this dataset.
        if cfg.classes is None:
            classes = list(range(MNIST_NUM_CLASSES))
        else:
            classes = [int(c) for c in cfg.classes]
            if not classes:
                raise ValueError("classes must be None or a non-empty tuple of ints")
            if any((c < 0 or c >= MNIST_NUM_CLASSES) for c in classes):
                raise ValueError(f"classes must be in [0, {MNIST_NUM_CLASSES - 1}]")
            if len(set(classes)) != len(classes):
                raise ValueError("classes must not contain duplicates")
        self.classes: tuple[int, ...] = tuple(classes)

        # Cache stores base images in [0,1] space (pre-normalization), so that
        # augmentations (if enabled) can be applied on-the-fly.
        self._cache_01: torch.Tensor | None = None
        if bool(cfg.cache_images):
            self._cache_01 = self._generate_all_01()

    def __len__(self) -> int:
        return len(self.classes) * self.images_per_class

    def _postprocess(self, x01: torch.Tensor) -> torch.Tensor:
        if bool(self.cfg.normalize_like_mnist):
            return (x01 - MNIST_MEAN) / MNIST_STD
        return x01

    def _format_label(self, cls: int):
        if self.cfg.label_format == "int":
            return int(cls)
        y = F.one_hot(torch.as_tensor(int(cls), dtype=torch.long), num_classes=MNIST_NUM_CLASSES)
        return y.to(torch.float32)

    def _generate_one_01(self, cls: int, j: int) -> torch.Tensor:
        g = torch.Generator()
        # Deterministic per-(cls, j) sampling.
        g.manual_seed(int(self.cfg.seed) + cls * 1_000_003 + j * 10_007)
        x = (torch.rand(MNIST_IMAGE_SHAPE, generator=g, dtype=torch.float32) < self.p_on).to(torch.float32)
        return x

    def _generate_all_01(self) -> torch.Tensor:
        imgs = torch.empty(
            (len(self.classes), self.images_per_class, *MNIST_IMAGE_SHAPE),
            dtype=torch.float32,
        )
        for i, cls in enumerate(self.classes):
            for j in range(self.images_per_class):
                imgs[i, j] = self._generate_one_01(int(cls), j)
        return imgs

    def _maybe_augment_01(self, x01: torch.Tensor, *, cls: int, j: int) -> torch.Tensor:
        if not bool(self.cfg.augment):
            return x01

        # Epoch-dependent per-sample augmentation.
        g = torch.Generator()
        g.manual_seed(int(self.cfg.seed) + cls * 1_000_003 + j * 10_007 + 777_777 + int(self.epoch) * 2_000_033)

        out = x01

        m = int(self.cfg.augment_jitter_max)
        if m > 0:
            dx = int(torch.randint(-m, m + 1, (1,), generator=g).item())
            dy = int(torch.randint(-m, m + 1, (1,), generator=g).item())
            if dx == 0 and dy == 0:
                dx = 1 if m >= 1 else 0
            if dx != 0 or dy != 0:
                out = torch.roll(out, shifts=(dy, dx), dims=(1, 2))

        std = float(self.cfg.augment_gaussian_std)
        if std > 0.0:
            n = torch.randn(out.shape, generator=g, dtype=out.dtype) * std
            out = (out + n).clamp_(0.0, 1.0)

        return out

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        cls_i = int(idx // self.images_per_class)
        cls = int(self.classes[cls_i])
        j = int(idx % self.images_per_class)

        if self._cache_01 is not None:
            x01 = self._cache_01[cls_i, j].clone()
        else:
            x01 = self._generate_one_01(cls, j)

        x01 = self._maybe_augment_01(x01, cls=cls, j=j)
        x = self._postprocess(x01)
        y = self._format_label(cls)
        return x, y


def make_lowbit_tag_datasets(*, cfg: LowBitTagConfig) -> tuple[Dataset, Dataset]:
    train_ds = LowBitTagDataset(cfg)
    test_seed = int(cfg.seed) if bool(cfg.test_split_same_images) else int(cfg.seed) + 1
    test_cfg = replace(cfg, seed=test_seed)
    if bool(cfg.augment) and not bool(cfg.augment_apply_to_test):
        test_cfg = replace(test_cfg, augment=False, augment_jitter_max=0, augment_gaussian_std=0.0)
    test_ds = LowBitTagDataset(test_cfg)
    return train_ds, test_ds


def make_lowbit_tag_loaders(
    *,
    cfg: LowBitTagConfig,
    batch_size: int,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds, test_ds = make_lowbit_tag_datasets(cfg=cfg)

    effective_pin_memory = bool(pin_memory)
    if device is not None:
        effective_pin_memory = effective_pin_memory and device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle_train),
        num_workers=int(num_workers),
        pin_memory=effective_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=effective_pin_memory,
    )
    return train_loader, test_loader


@dataclass(frozen=True)
class NoiseTagConfig:
    images_per_class: int = 200
    seed: int = 0
    normalize_like_mnist: bool = True
    distribution: NoiseDistribution = "uniform"
    cache_images: bool = True
    label_format: LabelFormat = "onehot"
    # Optional subset of MNIST digit-tags to generate. If None, uses all digits 0..9.
    # This is useful if you want to restrict the tag vocabulary, while still ensuring
    # train/test share the exact same set of tags.
    classes: tuple[int, ...] | None = None

    # Optional augmentation suite (applied in [0,1] space before normalization).
    # Defaults keep behavior identical to the pre-augmentation dataset.
    augment: bool = True
    # Cyclic shift (wrap-around) in pixels. 0 disables shifting.
    # Default 27 allows any non-trivial cyclic shift on 28x28 images.
    augment_shift_max: int = 27
    # Additional small Gaussian noise (std in [0,1] pixel space). 0 disables.
    augment_gaussian_std: float = 0.0
    # By default, augmentations are train-only; test uses clean samples.
    augment_apply_to_test: bool = False


class NoiseTagDataset(Dataset):
    """
    A synthetic "tag" dataset in the MNIST image domain.

    For each MNIST class c in {0..9}, create `images_per_class` random noise images.
    The label is the associated class c (int or one-hot).
    """

    def __init__(self, cfg: NoiseTagConfig):
        self.cfg = cfg
        self.epoch: int = 0
        self.images_per_class = int(cfg.images_per_class)
        if self.images_per_class <= 0:
            raise ValueError("images_per_class must be > 0")

        # Which digit-tags exist in this dataset.
        if cfg.classes is None:
            classes = list(range(MNIST_NUM_CLASSES))
        else:
            classes = [int(c) for c in cfg.classes]
            if not classes:
                raise ValueError("classes must be None or a non-empty tuple of ints")
            if any((c < 0 or c >= MNIST_NUM_CLASSES) for c in classes):
                raise ValueError(f"classes must be in [0, {MNIST_NUM_CLASSES - 1}]")
            if len(set(classes)) != len(classes):
                raise ValueError("classes must not contain duplicates")
        self.classes: tuple[int, ...] = tuple(classes)

        # Cache stores base images in [0,1] space (pre-normalization), so that
        # augmentations (if enabled) can be applied on-the-fly.
        self._cache_01: torch.Tensor | None = None
        if bool(cfg.cache_images):
            self._cache_01 = self._generate_all_01()

    def __len__(self) -> int:
        return len(self.classes) * self.images_per_class

    def _postprocess(self, x01: torch.Tensor) -> torch.Tensor:
        # x01 is expected in [0, 1] range.
        if bool(self.cfg.normalize_like_mnist):
            return (x01 - MNIST_MEAN) / MNIST_STD
        return x01

    def _sample_noise_01(self, *, generator: torch.Generator) -> torch.Tensor:
        if self.cfg.distribution == "uniform":
            return torch.rand(MNIST_IMAGE_SHAPE, generator=generator, dtype=torch.float32)
        if self.cfg.distribution == "normal":
            # Approximate MNIST-ish intensities; clamp to [0, 1]
            x = torch.randn(MNIST_IMAGE_SHAPE, generator=generator, dtype=torch.float32) * 0.2 + 0.5
            return x.clamp_(0.0, 1.0)
        raise ValueError(f"Unknown distribution: {self.cfg.distribution}")

    def _generate_one_01(self, cls: int, j: int) -> torch.Tensor:
        g = torch.Generator()
        # Deterministic per-(cls, j) sampling.
        g.manual_seed(int(self.cfg.seed) + cls * 1_000_003 + j * 10_007)
        return self._sample_noise_01(generator=g)

    def _generate_all_01(self) -> torch.Tensor:
        # Shape: (num_classes, images_per_class, 1, 28, 28) in [0,1].
        imgs = torch.empty(
            (len(self.classes), self.images_per_class, *MNIST_IMAGE_SHAPE),
            dtype=torch.float32,
        )
        for i, cls in enumerate(self.classes):
            for j in range(self.images_per_class):
                imgs[i, j] = self._generate_one_01(int(cls), j)
        return imgs

    def _format_label(self, cls: int):
        if self.cfg.label_format == "int":
            return int(cls)
        y = F.one_hot(torch.as_tensor(int(cls), dtype=torch.long), num_classes=MNIST_NUM_CLASSES)
        return y.to(torch.float32)

    def _maybe_augment_01(self, x01: torch.Tensor, *, cls: int, j: int) -> torch.Tensor:
        if not bool(self.cfg.augment):
            return x01

        # Epoch-dependent per-sample augmentation.
        g = torch.Generator()
        g.manual_seed(int(self.cfg.seed) + cls * 1_000_003 + j * 10_007 + 999_983 + int(self.epoch) * 2_000_033)

        out = x01

        m = int(self.cfg.augment_shift_max)
        if m > 0:
            dx = int(torch.randint(-m, m + 1, (1,), generator=g).item())
            dy = int(torch.randint(-m, m + 1, (1,), generator=g).item())
            # Ensure we actually shift (avoid the (0,0) no-op).
            if dx == 0 and dy == 0:
                dx = 1 if m >= 1 else 0
            if dx != 0 or dy != 0:
                # x01 shape is (1, H, W); roll dims (H, W) => dims (1, 2)
                out = torch.roll(out, shifts=(dy, dx), dims=(1, 2))

        std = float(self.cfg.augment_gaussian_std)
        if std > 0.0:
            n = torch.randn(out.shape, generator=g, dtype=out.dtype) * std
            out = (out + n).clamp_(0.0, 1.0)

        return out

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        cls_i = int(idx // self.images_per_class)
        cls = int(self.classes[cls_i])
        j = int(idx % self.images_per_class)
        if self._cache_01 is not None:
            x01 = self._cache_01[cls_i, j]
        else:
            x01 = self._generate_one_01(cls, j)

        # Clone if cached so we never mutate the cache in-place during augmentation.
        if self._cache_01 is not None:
            x01 = x01.clone()

        x01 = self._maybe_augment_01(x01, cls=cls, j=j)
        x = self._postprocess(x01)
        y = self._format_label(cls)
        return x, y


class CombinedDataset(Dataset):
    """
    Combine multiple datasets in the same input/output domain.

    For each item:
    - sample a (pseudo-)random element from each component dataset
    - sum images pixelwise
    - combine labels into a multi-hot vector via elementwise OR (implemented as clamp(sum,0,1))

    Assumptions:
    - all images have identical shape
    - all labels are either:
      - one-hot/multi-hot vectors of identical shape, or
      - ints in [0, num_classes), if `num_classes` is provided
    """

    def __init__(
        self,
        datasets: Sequence[Dataset | DataLoader],
        *,
        length: int | None = None,
        seed: int = 0,
        num_classes: int | None = None,
        clip: tuple[float, float] | None = None,
    ):
        if not datasets:
            raise ValueError("datasets must be a non-empty list")

        # Accept either raw Datasets or DataLoaders (useful when callers talk in terms
        # of "dataloaders", but we still combine at the dataset level).
        unwrapped: list[Dataset] = []
        for d in datasets:
            if isinstance(d, DataLoader):
                unwrapped.append(d.dataset)
            else:
                unwrapped.append(d)

        self.datasets = unwrapped
        self.seed = int(seed)
        self.epoch: int = 0
        self.num_classes = None if num_classes is None else int(num_classes)
        self.clip = clip

        if length is None:
            # Arbitrary but practical default: as long as the largest dataset.
            self.length = max(len(ds) for ds in self.datasets)
        else:
            self.length = int(length)
        if self.length <= 0:
            raise ValueError("length must be > 0")

    def __len__(self) -> int:
        return self.length

    def _rand_index(self, *, dataset_id: int, item_idx: int, ds_len: int) -> int:
        g = torch.Generator()
        g.manual_seed(self.seed + dataset_id * 1_000_003 + item_idx * 10_007 + int(self.epoch) * 2_000_033)
        return int(torch.randint(0, int(ds_len), (1,), generator=g).item())

    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch for epoch-dependent pairing.

        Also propagates to component datasets when they expose `set_epoch`.
        """
        self.epoch = int(epoch)
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                try:
                    ds.set_epoch(int(epoch))  # type: ignore[attr-defined]
                except Exception:
                    pass

    def _to_label_vec(self, y) -> torch.Tensor:
        if torch.is_tensor(y) and y.dtype.is_floating_point:
            if y.ndim != 1:
                raise ValueError(f"Expected 1D label vector, got shape={tuple(y.shape)}")
            return y.to(torch.float32)

        if torch.is_tensor(y) and y.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
            y_int = int(y.item())
        elif isinstance(y, int):
            y_int = int(y)
        else:
            raise ValueError(f"Unsupported label type: {type(y)}")

        if self.num_classes is None:
            raise ValueError("CombinedDataset got int labels but num_classes was not provided.")
        return F.one_hot(torch.as_tensor(y_int, dtype=torch.long), num_classes=self.num_classes).to(torch.float32)

    def __getitem__(self, idx: int):
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []

        for k, ds in enumerate(self.datasets):
            j = self._rand_index(dataset_id=k, item_idx=int(idx), ds_len=len(ds))
            x, y = ds[j]
            if not torch.is_tensor(x):
                raise ValueError(f"Dataset {k} returned non-tensor x: {type(x)}")
            xs.append(x)
            ys.append(self._to_label_vec(y))

        x0_shape = tuple(xs[0].shape)
        y0_shape = tuple(ys[0].shape)
        for k in range(1, len(xs)):
            if tuple(xs[k].shape) != x0_shape:
                raise ValueError(f"Image shape mismatch: ds0={x0_shape} dsk={tuple(xs[k].shape)}")
            if tuple(ys[k].shape) != y0_shape:
                raise ValueError(f"Label shape mismatch: ds0={y0_shape} dsk={tuple(ys[k].shape)}")

        x_sum = torch.stack(xs, dim=0).sum(dim=0)
        if self.clip is not None:
            lo, hi = float(self.clip[0]), float(self.clip[1])
            x_sum = x_sum.clamp(lo, hi)

        y_sum = torch.stack(ys, dim=0).sum(dim=0)
        y_mh = y_sum.clamp(0.0, 1.0)
        return x_sum, y_mh


def make_combined_datasets(
    *,
    data_dir: str | Path,
    noise_cfg: NoiseTagConfig,
    lowfreq_cfg: LowFreqTagConfig | None = None,
    lowbit_cfg: LowBitTagConfig | None = None,
    loop_cfg: LoopConfig | None = None,
    mnist_augment_cfg: MNISTAugmentConfig | None = None,
    mnist_normalize: bool = True,
    download: bool = True,
    seed: int = 0,
    length: int | None = None,
    clip: tuple[float, float] | None = None,
    datasets: Sequence[str] | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Create train/test CombinedDataset datasets from a list of component datasets.

    Supported component dataset tags:
    - "mnist"
    - "noise_tags"
    - "lowfreq_tag"
    - "lowbit_tag"
    - "loops"

    Output labels are multi-hot float vectors, suitable for multi-label training.
    """
    if datasets is None:
        # Backward-compatible default.
        component_tags = ["mnist", "noise_tags"]
    else:
        component_tags = [str(x) for x in datasets]
        if not component_tags:
            raise ValueError("combined.datasets must be null or a non-empty list of dataset tags")

    allowed = {"mnist", "noise_tags", "lowfreq_tag", "lowbit_tag", "loops"}
    unknown = [t for t in component_tags if t not in allowed]
    if unknown:
        raise ValueError(f"Unknown combined dataset tag(s): {unknown}. Allowed: {sorted(allowed)}")

    train_parts: list[Dataset] = []
    test_parts: list[Dataset] = []

    for tag in component_tags:
        if tag == "mnist":
            mnist_train, mnist_test = make_mnist_datasets(
                data_dir=data_dir,
                normalize=bool(mnist_normalize),
                label_format="onehot",
                download=bool(download),
                augment_cfg=mnist_augment_cfg,
            )
            train_parts.append(mnist_train)
            test_parts.append(mnist_test)
            continue

        if tag == "noise_tags":
            # Enforce one-hot for combined labels, even if the caller passed int.
            effective_noise_cfg = (
                noise_cfg if noise_cfg.label_format == "onehot" else replace(noise_cfg, label_format="onehot")
            )
            noise_train, noise_test = make_noise_tag_datasets(cfg=effective_noise_cfg)
            train_parts.append(noise_train)
            test_parts.append(noise_test)
            continue

        if tag == "lowfreq_tag":
            lf_cfg = lowfreq_cfg if lowfreq_cfg is not None else LowFreqTagConfig()
            # Enforce one-hot for combined labels.
            effective_lf_cfg = lf_cfg if lf_cfg.label_format == "onehot" else replace(lf_cfg, label_format="onehot")
            lf_train, lf_test = make_lowfreq_tag_datasets(data_dir=data_dir, cfg=effective_lf_cfg)
            train_parts.append(lf_train)
            test_parts.append(lf_test)
            continue

        if tag == "lowbit_tag":
            lb_cfg = lowbit_cfg if lowbit_cfg is not None else LowBitTagConfig()
            # Enforce one-hot for combined labels.
            effective_lb_cfg = lb_cfg if lb_cfg.label_format == "onehot" else replace(lb_cfg, label_format="onehot")
            lb_train, lb_test = make_lowbit_tag_datasets(cfg=effective_lb_cfg)
            train_parts.append(lb_train)
            test_parts.append(lb_test)
            continue

        if tag == "loops":
            l_cfg = loop_cfg if loop_cfg is not None else LoopConfig()
            # Enforce one-hot for combined labels (loops return an all-zeros vector).
            effective_l_cfg = l_cfg if l_cfg.label_format == "onehot" else replace(l_cfg, label_format="onehot")
            loop_train, loop_test = make_loop_datasets(cfg=effective_l_cfg)
            train_parts.append(loop_train)
            test_parts.append(loop_test)
            continue

        raise AssertionError("Unreachable")

    train_ds = CombinedDataset(train_parts, seed=int(seed), length=length, num_classes=None, clip=clip)
    test_ds = CombinedDataset(test_parts, seed=int(seed) + 1, length=length, num_classes=None, clip=clip)
    return train_ds, test_ds


def make_combined_loaders(
    *,
    data_dir: str | Path,
    noise_cfg: NoiseTagConfig,
    lowfreq_cfg: LowFreqTagConfig | None = None,
    lowbit_cfg: LowBitTagConfig | None = None,
    loop_cfg: LoopConfig | None = None,
    mnist_augment_cfg: MNISTAugmentConfig | None = None,
    batch_size: int,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    mnist_normalize: bool = True,
    download: bool = True,
    seed: int = 0,
    length: int | None = None,
    clip: tuple[float, float] | None = None,
    datasets: Sequence[str] | None = None,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds, test_ds = make_combined_datasets(
        data_dir=data_dir,
        noise_cfg=noise_cfg,
        lowfreq_cfg=lowfreq_cfg,
        lowbit_cfg=lowbit_cfg,
        loop_cfg=loop_cfg,
        mnist_augment_cfg=mnist_augment_cfg,
        mnist_normalize=mnist_normalize,
        download=download,
        seed=seed,
        length=length,
        clip=clip,
        datasets=datasets,
    )

    effective_pin_memory = bool(pin_memory)
    if device is not None:
        effective_pin_memory = effective_pin_memory and device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle_train),
        num_workers=int(num_workers),
        pin_memory=effective_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=effective_pin_memory,
    )
    return train_loader, test_loader


def denormalize_mnist(x: torch.Tensor) -> torch.Tensor:
    """Undo MNIST normalization (expects normalized MNIST-style tensors)."""
    return x * MNIST_STD + MNIST_MEAN

