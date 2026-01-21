from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

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


def mnist_transform(*, normalize: bool = True) -> transforms.Compose:
    if normalize:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            ]
        )
    return transforms.Compose([transforms.ToTensor()])


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
) -> tuple[Dataset, Dataset]:
    root = str(Path(data_dir))
    tfm = mnist_transform(normalize=bool(normalize))
    train_base = datasets.MNIST(root=root, train=True, download=bool(download), transform=tfm)
    test_base = datasets.MNIST(root=root, train=False, download=bool(download), transform=tfm)

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
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds, test_ds = make_mnist_datasets(
        data_dir=data_dir,
        normalize=normalize,
        label_format=label_format,
        download=download,
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


@dataclass(frozen=True)
class NoiseTagConfig:
    images_per_class: int = 256
    seed: int = 0
    normalize_like_mnist: bool = True
    distribution: NoiseDistribution = "uniform"
    cache_images: bool = True
    label_format: LabelFormat = "onehot"


class NoiseTagDataset(Dataset):
    """
    A synthetic "tag" dataset in the MNIST image domain.

    For each MNIST class c in {0..9}, create `images_per_class` random noise images.
    The label is the associated class c (int or one-hot).
    """

    def __init__(self, cfg: NoiseTagConfig):
        self.cfg = cfg
        self.images_per_class = int(cfg.images_per_class)
        if self.images_per_class <= 0:
            raise ValueError("images_per_class must be > 0")

        self._cache: torch.Tensor | None = None
        if bool(cfg.cache_images):
            self._cache = self._generate_all()

    def __len__(self) -> int:
        return MNIST_NUM_CLASSES * self.images_per_class

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

    def _generate_one(self, cls: int, j: int) -> torch.Tensor:
        g = torch.Generator()
        # Deterministic per-(cls, j) sampling.
        g.manual_seed(int(self.cfg.seed) + cls * 1_000_003 + j * 10_007)
        x01 = self._sample_noise_01(generator=g)
        return self._postprocess(x01)

    def _generate_all(self) -> torch.Tensor:
        # Shape: (10, images_per_class, 1, 28, 28)
        imgs = torch.empty(
            (MNIST_NUM_CLASSES, self.images_per_class, *MNIST_IMAGE_SHAPE),
            dtype=torch.float32,
        )
        for cls in range(MNIST_NUM_CLASSES):
            for j in range(self.images_per_class):
                imgs[cls, j] = self._generate_one(cls, j)
        return imgs

    def _format_label(self, cls: int):
        if self.cfg.label_format == "int":
            return int(cls)
        y = F.one_hot(torch.as_tensor(int(cls), dtype=torch.long), num_classes=MNIST_NUM_CLASSES)
        return y.to(torch.float32)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        cls = int(idx // self.images_per_class)
        j = int(idx % self.images_per_class)
        if self._cache is not None:
            x = self._cache[cls, j]
        else:
            x = self._generate_one(cls, j)
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
        g.manual_seed(self.seed + dataset_id * 1_000_003 + item_idx * 10_007)
        return int(torch.randint(0, int(ds_len), (1,), generator=g).item())

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


def denormalize_mnist(x: torch.Tensor) -> torch.Tensor:
    """Undo MNIST normalization (expects normalized MNIST-style tensors)."""
    return x * MNIST_STD + MNIST_MEAN

