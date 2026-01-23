"""
Train a simple VAE for 28x28 pixel reconstruction on:
- loop data (synthetic ring images)
- low-frequency tag data
- MNIST
- combined data (pixelwise sums, using CombinedDataset from dense_sparse_extractor/data.py)

Features:
- KL beta warmup + linear ramp
- Optional Weights & Biases logging with image/reconstruction grids

Run examples:
  python dense_sparse_extractor/experiments/combined_recon.py --dataset mnist
  python dense_sparse_extractor/experiments/combined_recon.py --dataset lowfreq_tag
  python dense_sparse_extractor/experiments/combined_recon.py --dataset loops --epochs 20
  python dense_sparse_extractor/experiments/combined_recon.py --dataset combined --combined_components mnist lowfreq_tag
  python dense_sparse_extractor/experiments/combined_recon.py --dataset mnist --disable_wandb
"""

from __future__ import annotations

import math
import os
import random
import sys
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid

try:
    from dense_sparse_extractor.data import (
        LoopConfig,
        LowBitTagConfig,
        LowFreqTagConfig,
        MNISTAugmentConfig,
        NoiseTagConfig,
        make_combined_datasets,
        make_lowbit_tag_datasets,
        make_lowfreq_tag_datasets,
        make_loop_datasets,
        make_mnist_datasets,
        make_noise_tag_datasets,
    )
except ModuleNotFoundError:
    # Allow running this file directly without `pip install -e .`
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from dense_sparse_extractor.data import (  # type: ignore[no-redef]
        LoopConfig,
        LowBitTagConfig,
        LowFreqTagConfig,
        MNISTAugmentConfig,
        NoiseTagConfig,
        make_combined_datasets,
        make_lowbit_tag_datasets,
        make_lowfreq_tag_datasets,
        make_loop_datasets,
        make_mnist_datasets,
        make_noise_tag_datasets,
    )


DatasetType = Literal["loops", "lowfreq_tag", "lowbit_tag", "mnist", "combined"]
ModelType = Literal["mlp_vae", "conv_vae", "mlp_sae", "mlp_topk_sae", "mlp_mp_sae", "combined_ae", "densae"]
CombinedSAEType = Literal["mlp_sae", "mlp_topk_sae"]


def _maybe_import_overcomplete():
    """
    Import Overcomplete from either site-packages or a sibling checkout.
    """
    try:
        import overcomplete  # type: ignore

        return overcomplete
    except ModuleNotFoundError:
        # If running from this repo checkout, `../overcomplete` is a common sibling.
        proj_root = Path(__file__).resolve().parents[1].parent  # .../projects
        sys.path.insert(0, str(proj_root))
        import overcomplete  # type: ignore

        return overcomplete


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available; using CPU.")
            return torch.device("cpu")
        return torch.device(device)
    raise ValueError(f"Unknown device: {device}")


def _safe_path_component(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    keep: list[str] = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    return out or "unnamed"


def _run_dir(*, checkpoint_root: Path, wandb_project: str, wandb_run_name: str) -> Path:
    return checkpoint_root / _safe_path_component(wandb_project) / _safe_path_component(wandb_run_name)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_checkpoint(
    *,
    run_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    tag: str | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"epoch_{int(epoch):04d}"
    if tag:
        stem = f"{stem}_{_safe_path_component(tag)}"
    ckpt_path = run_dir / f"{stem}.pt"
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, ckpt_path)
    # Always update latest.pt to the same payload (overwrite).
    torch.save(payload, run_dir / "latest.pt")
    return ckpt_path


class XOnlyDataset(Dataset):
    """Drop labels: (x,y) -> (x,x) or (x,0). Here we keep y dummy."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:  # pragma: no cover
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            try:
                self.base.set_epoch(int(epoch))  # type: ignore[attr-defined]
            except Exception:
                pass

    def __getitem__(self, idx: int):
        x, _y = self.base[idx]
        return x, 0


# -------------------------
# VAE
# -------------------------


class MLPVAE(nn.Module):
    def __init__(
        self,
        *,
        z_dim: int = 16,
        input_dim: int = 28 * 28,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        hidden_dim: int = 1024,
        n_layers: int = 3,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.input_dim = int(input_dim)
        self.image_shape = tuple(int(x) for x in image_shape)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        if self.n_layers <= 0:
            raise ValueError("n_layers must be >= 1")

        enc_layers: list[nn.Module] = []
        d = self.input_dim
        for _ in range(self.n_layers):
            enc_layers.append(nn.Linear(d, self.hidden_dim))
            enc_layers.append(nn.ReLU())
            d = self.hidden_dim
        self.enc = nn.Sequential(*enc_layers)
        self.enc_mu = nn.Linear(self.hidden_dim, self.z_dim)
        self.enc_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        dec_layers: list[nn.Module] = []
        d = self.z_dim
        for _ in range(self.n_layers):
            dec_layers.append(nn.Linear(d, self.hidden_dim))
            dec_layers.append(nn.ReLU())
            d = self.hidden_dim
        dec_layers.append(nn.Linear(self.hidden_dim, self.input_dim))
        self.dec = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_flat = x.view(x.shape[0], -1)
        if x_flat.shape[1] != self.input_dim:
            raise ValueError(f"Expected flattened dim {self.input_dim}, got {int(x_flat.shape[1])}")
        h = self.enc(x_flat)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h).clamp(min=-20.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        logits_flat = self.dec(z)
        c, h, w = self.image_shape
        return logits_flat.view(z.shape[0], int(c), int(h), int(w))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        return {"mu": mu, "logvar": logvar, "z": z, "recon_logits": logits}


class ConvVAE(nn.Module):
    def __init__(self, *, z_dim: int = 16, hidden_channels: int = 32):
        super().__init__()
        self.z_dim = int(z_dim)
        hc = int(hidden_channels)

        # Encoder: (B,1,28,28) -> (B, hc*2, 7, 7)
        self.enc = nn.Sequential(
            nn.Conv2d(1, hc, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(hc, hc * 2, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.ReLU(),
        )
        self.enc_fc = nn.Linear((hc * 2) * 7 * 7, hc * 4)
        self.enc_mu = nn.Linear(hc * 4, self.z_dim)
        self.enc_logvar = nn.Linear(hc * 4, self.z_dim)

        # Decoder: z -> (B,1,28,28) logits
        self.dec_fc = nn.Linear(self.z_dim, (hc * 2) * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hc * 2, hc, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(hc, 1, kernel_size=4, stride=2, padding=1),  # 28x28
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        h = h.flatten(1)
        h = F.relu(self.enc_fc(h))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h).clamp(min=-20.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec_fc(z))
        h = h.view(z.shape[0], -1, 7, 7)
        return self.dec(h)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        return {"mu": mu, "logvar": logvar, "z": z, "recon_logits": logits}


class CombinedAE(nn.Module):
    """
    Combine an MLP VAE with an (MLP) SAE/TopK-SAE by summing reconstruction logits:
        recon_logits(x) = recon_logits_vae(x) + recon_logits_sae(x)
    """

    def __init__(self, *, vae: MLPVAE, sae: nn.Module):
        super().__init__()
        self.vae = vae
        self.sae = sae
        # When False, SAE contribution is bypassed (and training loop can freeze SAE params).
        self.sae_enabled: bool = True
        self.z_dim = int(getattr(vae, "z_dim"))

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        # Delegate prior samples to the VAE component only.
        return self.vae.decode_logits(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        vae_out = self.vae(x)
        vae_logits = vae_out["recon_logits"]

        if not bool(getattr(self, "sae_enabled", True)):
            # SAE disabled: behave like the VAE-only model for recon.
            return {
                "recon_logits": vae_logits,
                "mu": vae_out["mu"],
                "logvar": vae_out["logvar"],
                "vae_recon_logits": vae_logits,
            }

        x_flat = x.view(x.shape[0], -1)
        z_pre, z, x_hat_flat = self.sae(x_flat)  # type: ignore[misc]
        c, h, w = getattr(self.vae, "image_shape", (1, 28, 28))
        sae_logits = x_hat_flat.view(x.shape[0], int(c), int(h), int(w))

        combined_logits = vae_logits + sae_logits
        return {
            "recon_logits": combined_logits,
            "mu": vae_out["mu"],
            "logvar": vae_out["logvar"],
            "vae_recon_logits": vae_logits,
            "sae_recon_logits": sae_logits,
            "sae_z_pre": z_pre,
            "sae_z": z,
        }


def kl_diag_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # Average over batch (sum over dims)
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1).mean()


@dataclass(frozen=True)
class KLBetaSchedule:
    beta_max: float = 1.0
    warmup_steps: int = 0
    ramp_steps: int = 10_000

    def beta(self, step: int) -> float:
        s = int(step)
        if s < int(self.warmup_steps):
            return 0.0
        if int(self.ramp_steps) <= 0:
            return float(self.beta_max)
        t = (s - int(self.warmup_steps)) / float(self.ramp_steps)
        t = max(0.0, min(1.0, t))
        return float(self.beta_max) * float(t)


# -------------------------
# Data builders
# -------------------------


def _make_recon_loaders(
    *,
    dataset: DatasetType,
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
    seed: int,
    loop_cfg: LoopConfig,
    mnist_augment: bool,
    mnist_brightness_clip_min: float,
    mnist_max_brightness: float,
    mnist_jitter_crop: int,
    mnist_rotation_degrees: float,
    mnist_gaussian_std: float,
    mnist_augment_apply_to_test: bool,
    lowfreq_images_per_class: int,
    lowfreq_classes: Sequence[int] | None,
    lowfreq_augment: bool,
    lowfreq_augment_jitter_max: int,
    lowfreq_augment_gaussian_std: float,
    lowfreq_augment_apply_to_test: bool,
    lowfreq_boost: float,
    lowbit_images_per_class: int,
    lowbit_classes: Sequence[int] | None,
    lowbit_augment: bool,
    lowbit_augment_jitter_max: int,
    lowbit_augment_gaussian_std: float,
    lowbit_augment_apply_to_test: bool,
    lowbit_p_on: float,
    lowbit_on_min_brightness: float,
    combined_components: Sequence[str],
    combined_space_separation: bool,
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)

    if dataset == "loops":
        train_base, test_base = make_loop_datasets(cfg=loop_cfg)
        train_ds = XOnlyDataset(train_base)
        test_ds = XOnlyDataset(test_base)
    elif dataset == "mnist":
        mnist_aug = None
        if bool(mnist_augment) or float(mnist_max_brightness) < 1.0:
            mnist_aug = MNISTAugmentConfig(
                augment=bool(mnist_augment),
                brightness_clip_min=float(mnist_brightness_clip_min),
                max_brightness=float(mnist_max_brightness),
                jitter_crop=int(mnist_jitter_crop),
                rotation_degrees=float(mnist_rotation_degrees),
                gaussian_std=float(mnist_gaussian_std),
                augment_apply_to_test=bool(mnist_augment_apply_to_test),
            )
        train_base, test_base = make_mnist_datasets(
            data_dir=data_dir,
            normalize=False,  # recon in [0,1]
            label_format="int",
            download=True,
            augment_cfg=mnist_aug,
        )
        train_ds = XOnlyDataset(train_base)
        test_ds = XOnlyDataset(test_base)
    elif dataset == "lowfreq_tag":
        lf_cfg = LowFreqTagConfig(
            images_per_class=int(lowfreq_images_per_class),
            seed=int(seed),
            test_split_same_images=False,
            normalize_like_mnist=False,  # recon in [0,1]
            cache_images=True,
            label_format="onehot",
            classes=tuple(int(c) for c in lowfreq_classes) if lowfreq_classes is not None else None,
            lowfreq_boost=float(lowfreq_boost),
            augment=bool(lowfreq_augment),
            augment_jitter_max=int(lowfreq_augment_jitter_max),
            augment_gaussian_std=float(lowfreq_augment_gaussian_std),
            augment_apply_to_test=bool(lowfreq_augment_apply_to_test),
        )
        train_base, test_base = make_lowfreq_tag_datasets(data_dir=data_dir, cfg=lf_cfg)
        train_ds = XOnlyDataset(train_base)
        test_ds = XOnlyDataset(test_base)
    elif dataset == "lowbit_tag":
        lb_cfg = LowBitTagConfig(
            images_per_class=int(lowbit_images_per_class),
            seed=int(seed),
            test_split_same_images=False,
            normalize_like_mnist=False,  # recon in [0,1]
            cache_images=True,
            label_format="onehot",
            classes=tuple(int(c) for c in lowbit_classes) if lowbit_classes is not None else None,
            p_on=float(lowbit_p_on),
            on_min_brightness=float(lowbit_on_min_brightness),
            augment=bool(lowbit_augment),
            augment_jitter_max=int(lowbit_augment_jitter_max),
            augment_gaussian_std=float(lowbit_augment_gaussian_std),
            augment_apply_to_test=bool(lowbit_augment_apply_to_test),
        )
        train_base, test_base = make_lowbit_tag_datasets(cfg=lb_cfg)
        train_ds = XOnlyDataset(train_base)
        test_ds = XOnlyDataset(test_base)
    elif dataset == "combined":
        # Default "combined" here means combining whichever components you request.
        # All components are kept in [0,1] and the sum is clipped to [0,1].
        noise_cfg = NoiseTagConfig(
            seed=int(seed),
            normalize_like_mnist=False,
            cache_images=True,
            label_format="onehot",
        )
        mnist_aug = None
        if bool(mnist_augment) or float(mnist_max_brightness) < 1.0:
            mnist_aug = MNISTAugmentConfig(
                augment=bool(mnist_augment),
                brightness_clip_min=float(mnist_brightness_clip_min),
                max_brightness=float(mnist_max_brightness),
                jitter_crop=int(mnist_jitter_crop),
                rotation_degrees=float(mnist_rotation_degrees),
                gaussian_std=float(mnist_gaussian_std),
                augment_apply_to_test=bool(mnist_augment_apply_to_test),
            )
        lowfreq_cfg = LowFreqTagConfig(
            images_per_class=int(lowfreq_images_per_class),
            seed=int(seed),
            test_split_same_images=False,
            normalize_like_mnist=False,
            cache_images=True,
            label_format="onehot",
            classes=tuple(int(c) for c in lowfreq_classes) if lowfreq_classes is not None else None,
            lowfreq_boost=float(lowfreq_boost),
            augment=bool(lowfreq_augment),
            augment_jitter_max=int(lowfreq_augment_jitter_max),
            augment_gaussian_std=float(lowfreq_augment_gaussian_std),
            augment_apply_to_test=bool(lowfreq_augment_apply_to_test),
        )
        lowbit_cfg = LowBitTagConfig(
            images_per_class=int(lowbit_images_per_class),
            seed=int(seed),
            test_split_same_images=False,
            normalize_like_mnist=False,
            cache_images=True,
            label_format="onehot",
            classes=tuple(int(c) for c in lowbit_classes) if lowbit_classes is not None else None,
            p_on=float(lowbit_p_on),
            on_min_brightness=float(lowbit_on_min_brightness),
            augment=bool(lowbit_augment),
            augment_jitter_max=int(lowbit_augment_jitter_max),
            augment_gaussian_std=float(lowbit_augment_gaussian_std),
            augment_apply_to_test=bool(lowbit_augment_apply_to_test),
        )
        train_base, test_base = make_combined_datasets(
            data_dir=data_dir,
            noise_cfg=noise_cfg,
            lowfreq_cfg=lowfreq_cfg,
            lowbit_cfg=lowbit_cfg,
            loop_cfg=loop_cfg,
            mnist_augment_cfg=mnist_aug,
            mnist_normalize=False,
            download=True,
            seed=int(seed),
            length=None,
            clip=(0.0, 1.0),
            datasets=list(combined_components),
            space_separation=bool(combined_space_separation),
        )
        train_ds = XOnlyDataset(train_base)
        test_ds = XOnlyDataset(test_base)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    effective_pin = bool(pin_memory) and device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=effective_pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=effective_pin,
    )
    return train_loader, test_loader


# -------------------------
# Training
# -------------------------


@torch.no_grad()
def _log_recon_images(
    *,
    wandb_run,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    global_step: int,
    n_images: int = 8,
    tag: str = "recon",
    log_prior_samples: bool = True,
    recon_loss: Literal["bce", "mse"] = "bce",
    image_shape: tuple[int, int, int] = (1, 28, 28),
    results: list[dict] | None = None,
) -> None:
    import wandb  # type: ignore

    model.eval()
    # Use a fixed set of indices for stable visualization across epochs.
    ds = loader.dataset
    n = min(int(n_images), len(ds))
    n = max(1, n)
    xs: list[torch.Tensor] = []
    for i in range(n):
        x_i, _y_i = ds[i]
        xs.append(x_i)
    x = torch.stack(xs, dim=0).to(device)
    # SAEs expect 2D inputs (B, D), while VAEs expect images (B,1,28,28).
    x_in = x
    if x_in.ndim > 2 and not hasattr(model, "decode_logits"):
        x_in = x_in.view(x_in.shape[0], -1)

    out = model(x_in)

    # Support both:
    # - VAE-style dict with "recon_logits"
    # - SAE-style tuple (z_pre, z, x_hat_flat)
    if isinstance(out, dict) and "recon_logits" in out:
        # VAE decoders always produce logits; even when training with MSE we interpret
        # recon as sigmoid(logits) in [0,1].
        x_hat = torch.sigmoid(out["recon_logits"])
    elif isinstance(out, (tuple, list)) and len(out) == 3:
        x_hat_flat = out[2]
        if not torch.is_tensor(x_hat_flat):
            raise ValueError("Unexpected SAE output type")
        c, h, w = tuple(int(x) for x in image_shape)
        x_hat_img = x_hat_flat.view(x_hat_flat.shape[0], c, h, w)
        if recon_loss == "bce":
            # For BCE training, interpret SAE-style outputs as logits.
            x_hat = torch.sigmoid(x_hat_img)
        else:
            # For MSE training, interpret SAE-style outputs as pixels in [0,1].
            x_hat = x_hat_img.clamp(0.0, 1.0)
    else:
        raise ValueError("Unsupported model output format for recon logging")

    def _to_rgb(img: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) -> (B,3,H,W) in [0,1]
        if img.ndim != 4:
            raise ValueError(f"Expected 4D image batch, got shape={tuple(img.shape)}")
        c = int(img.shape[1])
        if c == 3:
            return img
        if c == 1:
            return img.repeat(1, 3, 1, 1)
        # Fallback: take first channel as grayscale
        return img[:, :1].repeat(1, 3, 1, 1)

    # Diff visualization:
    # - white for 0
    # - red where original > recon
    # - blue where recon > original
    # Assumes x, x_hat in [0,1], so diff in [-1,1].
    diff = (x - x_hat).clamp(-1.0, 1.0)
    a = diff.abs().clamp(0.0, 1.0)
    pos = diff >= 0
    diff_rgb = torch.ones((diff.shape[0], 3, diff.shape[2], diff.shape[3]), device=diff.device, dtype=diff.dtype)
    # Positive: (1, 1-a, 1-a) => white -> red
    diff_rgb[:, 1] = torch.where(pos[:, 0], 1.0 - a[:, 0], diff_rgb[:, 1])
    diff_rgb[:, 2] = torch.where(pos[:, 0], 1.0 - a[:, 0], diff_rgb[:, 2])
    # Negative: (1-a, 1-a, 1) => white -> blue
    diff_rgb[:, 0] = torch.where(~pos[:, 0], 1.0 - a[:, 0], diff_rgb[:, 0])
    diff_rgb[:, 1] = torch.where(~pos[:, 0], 1.0 - a[:, 0], diff_rgb[:, 1])

    x_rgb = _to_rgb(x)
    x_hat_rgb = _to_rgb(x_hat)

    # Layout requested:
    # - 8 samples total, 4 columns
    # - row1: originals[0:4]
    # - row2: recons[0:4]
    # - row3: diffs[0:4]
    # - (larger vertical gap)
    # - row4: originals[4:8]
    # - row5: recons[4:8]
    # - row6: diffs[4:8]
    #
    # We implement this as two separate 3-row grids with a custom gap between them.
    n = min(int(n_images), int(x.shape[0]))
    n = max(1, n)
    n0 = min(4, n)
    n1 = max(0, n - n0)

    pad_value = 1.0
    pad = 2
    ncol = 4

    grid_top = make_grid(
        torch.cat([x_rgb[:n0], x_hat_rgb[:n0], diff_rgb[:n0]], dim=0),
        nrow=ncol,
        pad_value=pad_value,
        padding=pad,
    )
    if n1 > 0:
        grid_bot = make_grid(
            torch.cat([x_rgb[n0 : n0 + n1], x_hat_rgb[n0 : n0 + n1], diff_rgb[n0 : n0 + n1]], dim=0),
            nrow=ncol,
            pad_value=pad_value,
            padding=pad,
        )
    else:
        grid_bot = None

    if grid_bot is not None:
        # Larger gap between the two blocks.
        gap_px = 12
        gap = torch.full((grid_top.shape[0], gap_px, grid_top.shape[2]), pad_value, dtype=grid_top.dtype, device=grid_top.device)
        grid = torch.cat([grid_top, gap, grid_bot], dim=1)
    else:
        grid = grid_top

    wandb_run.log({f"images/{tag}": wandb.Image(grid), "step": int(global_step), "global_step": int(global_step)}, step=int(global_step))
    if results is not None:
        results.append({"step": int(global_step), "global_step": int(global_step), f"images/{tag}": "<image>"})

    # For VAEs, optionally log a few random prior samples (SAEs don't have a meaningful prior).
    if bool(log_prior_samples) and hasattr(model, "decode_logits") and hasattr(model, "z_dim"):
        z_dim = int(getattr(model, "z_dim"))
        z = torch.randn(int(n_images), z_dim, device=device)
        decode_logits = getattr(model, "decode_logits")
        samples = torch.sigmoid(decode_logits(z))
        grid_s = make_grid(samples[: int(n_images)], nrow=4, pad_value=pad_value, padding=pad)
        wandb_run.log(
            {f"images/{tag}_prior_samples": wandb.Image(grid_s), "step": int(global_step), "global_step": int(global_step)},
            step=int(global_step),
        )
        if results is not None:
            results.append(
                {
                    "step": int(global_step),
                    "global_step": int(global_step),
                    f"images/{tag}_prior_samples": "<image>",
                }
            )


def train(
    *,
    dataset: DatasetType,
    data_dir: str | Path,
    combined_components: Sequence[str],
    combined_space_separation: bool,
    combined_tag_weight: float = 1.0,
    combined_tag_pos_weight: float = 1.0,
    combined_partition_experts: bool = False,
    pos_weight: float = 1.0,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    lr: float,
    weight_decay: float,
    model_type: ModelType,
    z_dim: int,
    hidden_channels: int,
    mlp_hidden_dim: int,
    mlp_layers: int,
    sae_n_concepts: int,
    sae_l1_penalty: float,
    topk_k: int,
    mp_k: int,
    mp_dropout: float | None,
    sae_encoder_hidden_dim: int,
    sae_encoder_layers: int,
    combined_sae_type: CombinedSAEType = "mlp_topk_sae",
    dead_sae_epochs: int = 0,
    densae_n_dense: int = 256,
    densae_n_sparse: int = 2048,
    densae_iters: int = 10,
    densae_step_x: float = 1e-2,
    densae_step_u: float = 1e-2,
    densae_lambda_x: float = 0.0,
    densae_lambda_u: float = 1e-2,
    densae_fista: bool = True,
    densae_twosided_u: bool = True,
    mnist_augment: bool,
    mnist_brightness_clip_min: float,
    mnist_max_brightness: float,
    mnist_jitter_crop: int,
    mnist_rotation_degrees: float,
    mnist_gaussian_std: float,
    mnist_augment_apply_to_test: bool,
    recon_loss: Literal["bce", "mse"],
    kl_sched: KLBetaSchedule,
    log_every_steps: int,
    eval_every_epochs: int,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_run_name: str | None,
    lowfreq_images_per_class: int,
    lowfreq_classes: Sequence[int] | None,
    lowfreq_augment: bool,
    lowfreq_augment_jitter_max: int,
    lowfreq_augment_gaussian_std: float,
    lowfreq_augment_apply_to_test: bool,
    lowfreq_boost: float,
    lowbit_images_per_class: int,
    lowbit_classes: Sequence[int] | None,
    lowbit_augment: bool,
    lowbit_augment_jitter_max: int,
    lowbit_augment_gaussian_std: float,
    lowbit_augment_apply_to_test: bool,
    lowbit_p_on: float,
    lowbit_on_min_brightness: float,
    loop_cfg: LoopConfig,
    checkpoint_root: str | Path,
    checkpoint_interval: int,
) -> None:
    _set_seed(int(seed))

    train_loader, test_loader = _make_recon_loaders(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        device=device,
        seed=seed,
        loop_cfg=loop_cfg,
        mnist_augment=mnist_augment,
        mnist_brightness_clip_min=mnist_brightness_clip_min,
        mnist_max_brightness=mnist_max_brightness,
        mnist_jitter_crop=mnist_jitter_crop,
        mnist_rotation_degrees=mnist_rotation_degrees,
        mnist_gaussian_std=mnist_gaussian_std,
        mnist_augment_apply_to_test=mnist_augment_apply_to_test,
        lowfreq_images_per_class=lowfreq_images_per_class,
        lowfreq_classes=lowfreq_classes,
        lowfreq_augment=lowfreq_augment,
        lowfreq_augment_jitter_max=lowfreq_augment_jitter_max,
        lowfreq_augment_gaussian_std=lowfreq_augment_gaussian_std,
        lowfreq_augment_apply_to_test=lowfreq_augment_apply_to_test,
        lowfreq_boost=lowfreq_boost,
        lowbit_images_per_class=lowbit_images_per_class,
        lowbit_classes=lowbit_classes,
        lowbit_augment=lowbit_augment,
        lowbit_augment_jitter_max=lowbit_augment_jitter_max,
        lowbit_augment_gaussian_std=lowbit_augment_gaussian_std,
        lowbit_augment_apply_to_test=lowbit_augment_apply_to_test,
        lowbit_p_on=lowbit_p_on,
        lowbit_on_min_brightness=lowbit_on_min_brightness,
        combined_components=combined_components,
        combined_space_separation=bool(combined_space_separation),
    )

    # Infer input dimensionality / image shape from the dataset so models match
    # (e.g. 28x28 vs 28x56 when combined_space_separation=True).
    x0, _y0 = train_loader.dataset[0]
    if not torch.is_tensor(x0):
        raise ValueError(f"Expected dataset x to be a torch.Tensor, got {type(x0)}")
    if x0.ndim != 3:
        raise ValueError(f"Expected image-shaped tensor (C,H,W), got shape={tuple(x0.shape)}")
    image_shape = (int(x0.shape[0]), int(x0.shape[1]), int(x0.shape[2]))
    input_dim = int(x0.numel())

    # For combined+space_separation, we can build per-component spatial slices to:
    # - upweight sparse-tag reconstruction
    # - (optionally) force VAE to own mnist slice while SAE owns tag slices
    component_slices: list[tuple[str, slice]] | None = None
    if dataset == "combined" and bool(combined_space_separation):
        n_comp = int(len(combined_components))
        if n_comp <= 0:
            raise ValueError("combined_components must be non-empty when dataset='combined'")
        _c, _h, w_total = image_shape
        if int(w_total) % n_comp != 0:
            raise ValueError(
                f"combined_space_separation expects width divisible by #components; got W={w_total}, n={n_comp}"
            )
        w_each = int(w_total) // n_comp
        component_slices = []
        for i, name in enumerate(list(combined_components)):
            component_slices.append((str(name), slice(int(i * w_each), int((i + 1) * w_each))))

    def _make_recon_weights(x_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Build (weight_mask, pos_weight) broadcastable to x_img.
        - weight_mask: applies to all losses (BCE/MSE), all pixels
        - pos_weight: only used for BCE, weights positive targets (x==1)

        Global `pos_weight` applies everywhere (no dataset/region checks).
        Tag-only weights apply only for combined+space_separation via component_slices.
        """
        wmask = torch.ones_like(x_img, dtype=torch.float32)
        posw: torch.Tensor | None = None

        pw = float(pos_weight)
        if (not math.isfinite(pw)) or pw <= 0.0:
            raise ValueError(f"pos_weight must be finite and > 0, got {pos_weight}")
        if pw != 1.0:
            posw = torch.ones_like(x_img, dtype=torch.float32) * pw

        if component_slices is None:
            return wmask, posw

        tag_w = float(combined_tag_weight)
        tag_pos_w = float(combined_tag_pos_weight)
        if (not math.isfinite(tag_w)) or tag_w <= 0.0:
            raise ValueError(f"combined_tag_weight must be finite and > 0, got {combined_tag_weight}")
        if (not math.isfinite(tag_pos_w)) or tag_pos_w <= 0.0:
            raise ValueError(f"combined_tag_pos_weight must be finite and > 0, got {combined_tag_pos_weight}")

        if tag_w != 1.0:
            for name, sl in component_slices:
                if name != "mnist":
                    wmask[..., sl] = wmask[..., sl] * tag_w

        if tag_pos_w != 1.0:
            if posw is None:
                posw = torch.ones_like(x_img, dtype=torch.float32)
            for name, sl in component_slices:
                if name != "mnist":
                    posw[..., sl] = posw[..., sl] * tag_pos_w
        return wmask, posw

    # --- checkpoint/run bookkeeping ---
    checkpoint_root = Path(checkpoint_root)
    effective_run_name = str(wandb_run_name) if wandb_run_name else f"{dataset}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = _run_dir(checkpoint_root=checkpoint_root, wandb_project=str(wandb_project), wandb_run_name=effective_run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "mlp_vae":
        model: nn.Module = MLPVAE(
            z_dim=int(z_dim),
            input_dim=int(input_dim),
            image_shape=image_shape,
            hidden_dim=int(mlp_hidden_dim),
            n_layers=int(mlp_layers),
        ).to(device)
    elif model_type == "conv_vae":
        if tuple(image_shape) != (1, 28, 28):
            raise ValueError(f"conv_vae currently requires image_shape=(1,28,28), got {image_shape}")
        model = ConvVAE(z_dim=int(z_dim), hidden_channels=int(hidden_channels)).to(device)
    elif model_type == "combined_ae":
        oc = _maybe_import_overcomplete()
        SAE = oc.sae.SAE  # type: ignore[attr-defined]
        TopKSAE = oc.sae.TopKSAE  # type: ignore[attr-defined]
        MLPEncoder = oc.sae.MLPEncoder  # type: ignore[attr-defined]

        vae = MLPVAE(
            z_dim=int(z_dim),
            input_dim=int(input_dim),
            image_shape=image_shape,
            hidden_dim=int(mlp_hidden_dim),
            n_layers=int(mlp_layers),
        ).to(device)

        enc = MLPEncoder(
            int(input_dim),
            int(sae_n_concepts),
            hidden_dim=int(sae_encoder_hidden_dim),
            nb_blocks=int(sae_encoder_layers),
            residual=False,
            device=str(device),
        )
        if combined_sae_type == "mlp_sae":
            sae = SAE(int(input_dim), int(sae_n_concepts), encoder_module=enc, device=str(device))
        else:
            sae = TopKSAE(int(input_dim), int(sae_n_concepts), top_k=int(topk_k), encoder_module=enc, device=str(device))

        model = CombinedAE(vae=vae, sae=sae).to(device)
    elif model_type in ("mlp_sae", "mlp_topk_sae", "mlp_mp_sae"):
        oc = _maybe_import_overcomplete()
        SAE = oc.sae.SAE  # type: ignore[attr-defined]
        TopKSAE = oc.sae.TopKSAE  # type: ignore[attr-defined]
        MpSAE = oc.sae.MpSAE  # type: ignore[attr-defined]
        MLPEncoder = oc.sae.MLPEncoder  # type: ignore[attr-defined]

        if model_type == "mlp_mp_sae":
            # MpSAE uses greedy matching pursuit over the dictionary to produce sparse codes.
            # It does not use an MLP encoder in its encode() implementation.
            model = MpSAE(
                int(input_dim),
                int(sae_n_concepts),
                k=int(mp_k),
                dropout=mp_dropout,
                device=str(device),
            )
        else:
            enc = MLPEncoder(
                int(input_dim),
                int(sae_n_concepts),
                hidden_dim=int(sae_encoder_hidden_dim),
                nb_blocks=int(sae_encoder_layers),
                residual=False,
                device=str(device),
            )
            if model_type == "mlp_sae":
                model = SAE(int(input_dim), int(sae_n_concepts), encoder_module=enc, device=str(device))
            else:
                model = TopKSAE(int(input_dim), int(sae_n_concepts), top_k=int(topk_k), encoder_module=enc, device=str(device))
    elif model_type == "densae":
        try:
            from dense_sparse_extractor.densae import DenSAE, DenSAEConfig
        except ModuleNotFoundError:
            # Allow running this file directly without `pip install -e .`
            repo_root = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(repo_root))
            from dense_sparse_extractor.densae import DenSAE, DenSAEConfig  # type: ignore[no-redef]

        densae_cfg = DenSAEConfig(
            input_dim=int(input_dim),
            n_dense=int(densae_n_dense),
            n_sparse=int(densae_n_sparse),
            n_iters=int(densae_iters),
            step_x=float(densae_step_x),
            step_u=float(densae_step_u),
            lambda_x=float(densae_lambda_x),
            lambda_u=float(densae_lambda_u),
            fista=bool(densae_fista),
            twosided_u=bool(densae_twosided_u),
            normalize_init=True,
        )
        model = DenSAE(densae_cfg, device=device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    def _set_requires_grad(m: nn.Module, enabled: bool) -> None:
        for p in m.parameters():
            p.requires_grad_(bool(enabled))

    def _scale_weight_params(m: nn.Module, scale: float) -> None:
        s = float(scale)
        if not math.isfinite(s) or s <= 0.0:
            raise ValueError(f"scale must be finite and > 0, got {scale}")
        with torch.no_grad():
            for _name, p in m.named_parameters():
                # Scale "weight-like" tensors; keep biases/1D params unchanged.
                if p.ndim >= 2:
                    p.mul_(s)

    # For combined_ae we optionally keep the SAE "dead" for a few epochs:
    # - bypass its contribution to the recon
    # - freeze its weights (no grads / no updates)
    dead_sae_epochs = int(dead_sae_epochs)
    if model_type == "combined_ae" and dead_sae_epochs > 0:
        assert isinstance(model, CombinedAE)
        model.sae_enabled = False
        _set_requires_grad(model.sae, False)

    # Save final resolved args/config next to checkpoints.
    args_snapshot = {
        "dataset": str(dataset),
        "data_dir": str(data_dir),
        "combined_components": list(combined_components),
        "combined_space_separation": bool(combined_space_separation),
        "combined_tag_weight": float(combined_tag_weight),
        "combined_tag_pos_weight": float(combined_tag_pos_weight),
        "combined_partition_experts": bool(combined_partition_experts),
        "pos_weight": float(pos_weight),
        "seed": int(seed),
        "device": str(device),
        "inferred_image_shape": list(image_shape),
        "inferred_input_dim": int(input_dim),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "model_type": str(model_type),
        "z_dim": int(z_dim),
        "hidden_channels": int(hidden_channels),
        "mlp_hidden_dim": int(mlp_hidden_dim),
        "mlp_layers": int(mlp_layers),
        "sae_n_concepts": int(sae_n_concepts),
        "sae_l1_penalty": float(sae_l1_penalty),
        "topk_k": int(topk_k),
        "mp_k": int(mp_k),
        "mp_dropout": float(mp_dropout) if mp_dropout is not None else None,
        "sae_encoder_hidden_dim": int(sae_encoder_hidden_dim),
        "sae_encoder_layers": int(sae_encoder_layers),
        "combined_sae_type": str(combined_sae_type),
        "dead_sae_epochs": int(dead_sae_epochs),
        "densae_n_dense": int(densae_n_dense),
        "densae_n_sparse": int(densae_n_sparse),
        "densae_iters": int(densae_iters),
        "densae_step_x": float(densae_step_x),
        "densae_step_u": float(densae_step_u),
        "densae_lambda_x": float(densae_lambda_x),
        "densae_lambda_u": float(densae_lambda_u),
        "densae_fista": bool(densae_fista),
        "densae_twosided_u": bool(densae_twosided_u),
        "mnist_augment": bool(mnist_augment),
        "mnist_brightness_clip_min": float(mnist_brightness_clip_min),
        "mnist_max_brightness": float(mnist_max_brightness),
        "mnist_jitter_crop": int(mnist_jitter_crop),
        "mnist_rotation_degrees": float(mnist_rotation_degrees),
        "mnist_gaussian_std": float(mnist_gaussian_std),
        "mnist_augment_apply_to_test": bool(mnist_augment_apply_to_test),
        "recon_loss": str(recon_loss),
        "kl_schedule": asdict(kl_sched),
        "lowfreq_images_per_class": int(lowfreq_images_per_class),
        "lowfreq_classes": list(lowfreq_classes) if lowfreq_classes is not None else None,
        "lowfreq_augment": bool(lowfreq_augment),
        "lowfreq_augment_jitter_max": int(lowfreq_augment_jitter_max),
        "lowfreq_augment_gaussian_std": float(lowfreq_augment_gaussian_std),
        "lowfreq_augment_apply_to_test": bool(lowfreq_augment_apply_to_test),
        "lowfreq_boost": float(lowfreq_boost),
        "lowbit_images_per_class": int(lowbit_images_per_class),
        "lowbit_classes": list(lowbit_classes) if lowbit_classes is not None else None,
        "lowbit_augment": bool(lowbit_augment),
        "lowbit_augment_jitter_max": int(lowbit_augment_jitter_max),
        "lowbit_augment_gaussian_std": float(lowbit_augment_gaussian_std),
        "lowbit_augment_apply_to_test": bool(lowbit_augment_apply_to_test),
        "lowbit_p_on": float(lowbit_p_on),
        "lowbit_on_min_brightness": float(lowbit_on_min_brightness),
        "loop_cfg": asdict(loop_cfg),
        "wandb_enabled": bool(wandb_enabled),
        "wandb_project": str(wandb_project),
        "wandb_run_name": str(wandb_run_name) if wandb_run_name else None,
        "effective_run_name": effective_run_name,
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_interval": int(checkpoint_interval),
    }
    _write_json(run_dir / "args.json", args_snapshot)

    # Save initial checkpoint (epoch 0, right after init).
    global_step = 0
    _save_checkpoint(run_dir=run_dir, model=model, optimizer=opt, epoch=0, global_step=global_step, tag="init")

    wandb_run = None
    if bool(wandb_enabled):
        try:
            import wandb  # type: ignore
        except Exception as e:
            raise RuntimeError("wandb_enabled=True but wandb is not installed.") from e

        wandb_run = wandb.init(
            project=str(wandb_project),
            name=effective_run_name,
            config={
                **args_snapshot,
            },
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    # Accumulate all logs (including those sent to W&B) for a local JSON record.
    results: list[dict] = []
    results_path = run_dir / "results.json"

    for epoch in range(1, int(epochs) + 1):
        # For combined_ae, optionally "turn on" the SAE after `dead_sae_epochs`.
        if model_type == "combined_ae":
            assert isinstance(model, CombinedAE)
            should_enable = int(epoch) > int(dead_sae_epochs)
            if should_enable and not bool(getattr(model, "sae_enabled", True)):
                # Enable without rescaling SAE weights.
                model.sae_enabled = True
                _set_requires_grad(model.sae, True)
                print(f"[combined_ae] enabling SAE at epoch {epoch} (dead_sae_epochs={dead_sae_epochs})", flush=True)
            elif (not should_enable) and bool(getattr(model, "sae_enabled", True)):
                model.sae_enabled = False
                _set_requires_grad(model.sae, False)

        # Ensure epoch-dependent datasets update.
        if hasattr(train_loader.dataset, "set_epoch"):
            try:
                train_loader.dataset.set_epoch(int(epoch))  # type: ignore[attr-defined]
            except Exception:
                pass

        model.train()
        # recon_mean = mean per pixel/element (comparable across SAE/VAE)
        sum_recon = 0.0
        # recon_per_image = sum over pixels, mean over batch (useful for intuition)
        sum_recon_per_image = 0.0
        sum_kl = 0.0
        sum_loss = 0.0
        sum_l0 = 0.0
        n_seen = 0
        alive_mask = None
        if model_type in ("mlp_sae", "mlp_topk_sae", "mlp_mp_sae"):
            alive_mask = torch.zeros(int(sae_n_concepts), dtype=torch.bool, device=device)
        elif model_type == "combined_ae" and bool(getattr(model, "sae_enabled", True)):
            alive_mask = torch.zeros(int(sae_n_concepts), dtype=torch.bool, device=device)
        elif model_type == "densae":
            alive_mask = torch.zeros(int(densae_n_sparse), dtype=torch.bool, device=device)

        for step_in_epoch, (x, _y) in enumerate(train_loader, start=1):
            x = x.to(device)
            if model_type in ("mlp_vae", "conv_vae", "combined_ae"):
                out = model(x)
            else:
                x_flat = x.view(x.shape[0], -1)
                z_pre, z, x_hat_flat = model(x_flat)  # type: ignore[misc]
                out = (z_pre, z, x_hat_flat)

            if model_type in ("mlp_vae", "conv_vae", "combined_ae"):
                logits = out["recon_logits"]  # type: ignore[index]
                if (
                    model_type == "combined_ae"
                    and bool(combined_partition_experts)
                    and component_slices is not None
                    and isinstance(out, dict)
                    and ("vae_recon_logits" in out)
                    and ("sae_recon_logits" in out)
                ):
                    # Force VAE to explain MNIST slice, SAE to explain tag slices.
                    logits_used = torch.zeros_like(logits)
                    vae_logits = out["vae_recon_logits"]  # type: ignore[index]
                    sae_logits = out["sae_recon_logits"]  # type: ignore[index]
                    for name, sl in component_slices:
                        src = vae_logits if name == "mnist" else sae_logits
                        logits_used[..., sl] = src[..., sl]
                    logits = logits_used

                wmask, posw = _make_recon_weights(x)
                if recon_loss == "bce":
                    elem = F.binary_cross_entropy_with_logits(logits, x, reduction="none", pos_weight=posw)
                    weighted = elem * wmask
                    recon_mean = weighted.sum() / wmask.sum().clamp_min(1.0)
                    recon_per_image = weighted.view(weighted.shape[0], -1).sum(dim=1).mean()
                elif recon_loss == "mse":
                    x_hat = torch.sigmoid(logits)
                    elem = (x_hat - x).pow(2)
                    weighted = elem * wmask
                    recon_mean = weighted.sum() / wmask.sum().clamp_min(1.0)
                    recon_per_image = weighted.view(weighted.shape[0], -1).sum(dim=1).mean()
                else:
                    raise ValueError(f"Unknown recon_loss: {recon_loss}")

                kl = kl_diag_gaussian(out["mu"], out["logvar"])  # type: ignore[index]
                beta = kl_sched.beta(global_step)
                # Keep the *optimization* loss scale as "per-image" (compatible with old runs).
                sae_reg = torch.zeros((), device=device)
                l0 = float("nan")
                if model_type == "combined_ae" and bool(getattr(model, "sae_enabled", True)):
                    z = out["sae_z"]  # type: ignore[index]
                    active = (z.abs() > 0)
                    l0_t = active.float().sum(dim=1).mean()
                    sum_l0 += float(l0_t.item()) * int(x.shape[0])
                    l0 = float(l0_t.item())
                    if alive_mask is not None:
                        alive_mask |= active.any(dim=0)

                    if combined_sae_type == "mlp_sae":
                        codes_l1 = z.abs().sum(dim=1).mean()
                        sae_reg = float(sae_l1_penalty) * codes_l1

                loss = recon_per_image + (float(beta) * kl) + sae_reg
            else:
                z_pre, z, x_hat_flat = out  # type: ignore[misc]
                wmask_img, posw_img = _make_recon_weights(x)
                wmask_flat = wmask_img.view(wmask_img.shape[0], -1)
                posw_flat = None if posw_img is None else posw_img.view(posw_img.shape[0], -1)
                if recon_loss == "bce":
                    elem = F.binary_cross_entropy_with_logits(x_hat_flat, x_flat, reduction="none", pos_weight=posw_flat)
                    weighted = elem * wmask_flat
                    recon_mean = weighted.sum() / wmask_flat.sum().clamp_min(1.0)
                    recon_per_image = weighted.sum(dim=1).mean()
                else:
                    elem = (x_hat_flat - x_flat).pow(2)
                    weighted = elem * wmask_flat
                    recon_mean = weighted.sum() / wmask_flat.sum().clamp_min(1.0)
                    recon_per_image = weighted.sum(dim=1).mean()

                # MpSAE codes can be negative; track sparsity by nonzero entries.
                active = (z.abs() > 0)
                l0 = active.float().sum(dim=1).mean()
                sum_l0 += float(l0.item()) * int(x.shape[0])
                if alive_mask is not None:
                    alive_mask |= active.any(dim=0)

                # SAE/DenSAE loss: add optional L1 penalty for vanilla SAE (and for DenSAE
                # to prevent trivial "dense-u" solutions via dictionary scaling).
                if model_type == "mlp_sae":
                    codes_l1 = z.abs().sum(dim=1).mean()
                    loss = recon_mean + (float(sae_l1_penalty) * codes_l1)
                elif model_type == "densae":
                    codes_l1 = z.abs().sum(dim=1).mean()
                    loss = recon_mean + (float(densae_lambda_u) * codes_l1)
                else:
                    loss = recon_mean

                kl = torch.zeros((), device=device)
                beta = 0.0

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if model_type == "densae" and hasattr(model, "normalize"):
                # Enforce column normalization so sparsity threshold can't be bypassed by
                # simply scaling up the dictionary.
                try:
                    model.normalize()  # type: ignore[call-arg]
                except Exception:
                    pass

            bs = int(x.shape[0])
            n_seen += bs
            sum_recon += float(recon_mean.item()) * bs
            sum_recon_per_image += float(recon_per_image) * bs
            sum_kl += float(kl.item()) * bs
            sum_loss += float(loss.item()) * bs

            global_step += 1

            if wandb_run is not None and int(log_every_steps) > 0 and (global_step % int(log_every_steps) == 0):
                alive_count = float("nan")
                alive_pct = float("nan")
                dead_count = float("nan")
                dead_pct = float("nan")
                if alive_mask is not None:
                    total = float(int(densae_n_sparse if model_type == "densae" else sae_n_concepts))
                    alive_count = float(alive_mask.sum().item())
                    alive_pct = float(alive_count / max(1.0, total))
                    dead_count = float(total - alive_count)
                    dead_pct = float(1.0 - alive_pct)
                log_payload = {
                        "epoch": int(epoch),
                        "step": int(global_step),
                        "global_step": int(global_step),
                        "train/loss": float(sum_loss / max(1, n_seen)),
                        "train/recon": float(sum_recon / max(1, n_seen)),
                        "train/recon_per_image": float(sum_recon_per_image / max(1, n_seen)),
                        "train/kl": float(sum_kl / max(1, n_seen)),
                        "train/kl_beta": float(beta),
                        "train/avg_l0": float(sum_l0 / max(1, n_seen))
                        if model_type in ("mlp_sae", "mlp_topk_sae", "mlp_mp_sae", "combined_ae", "densae")
                        else float("nan"),
                        # Derived consistently from alive_count.
                        "train/alive_features": float(alive_count),
                        "train/alive_features_pct": float(alive_pct),
                        "train/dead_features": float(dead_pct),
                        "train/dead_features_count": float(dead_count),
                }
                results.append(log_payload)
                _write_json(results_path, results)
                if wandb_run is not None:
                    wandb_run.log(log_payload, step=int(global_step))

        # Evaluate
        if int(eval_every_epochs) > 0 and (epoch % int(eval_every_epochs) == 0):
            model.eval()
            t_sum_recon = 0.0
            t_sum_recon_per_image = 0.0
            t_sum_kl = 0.0
            t_sum_loss = 0.0
            t_sum_l0 = 0.0
            t_seen = 0
            t_alive_mask = None
            if model_type in ("mlp_sae", "mlp_topk_sae", "mlp_mp_sae"):
                t_alive_mask = torch.zeros(int(sae_n_concepts), dtype=torch.bool, device=device)
            elif model_type == "combined_ae" and bool(getattr(model, "sae_enabled", True)):
                t_alive_mask = torch.zeros(int(sae_n_concepts), dtype=torch.bool, device=device)
            elif model_type == "densae":
                t_alive_mask = torch.zeros(int(densae_n_sparse), dtype=torch.bool, device=device)
            with torch.no_grad():
                for x, _y in test_loader:
                    x = x.to(device)
                    if model_type in ("mlp_vae", "conv_vae", "combined_ae"):
                        out = model(x)
                        if recon_loss == "bce":
                            recon_mean = F.binary_cross_entropy_with_logits(out["recon_logits"], x, reduction="mean")  # type: ignore[index]
                            recon_per_image = (
                                F.binary_cross_entropy_with_logits(out["recon_logits"], x, reduction="sum") / x.shape[0]  # type: ignore[index]
                            )
                        else:
                            x_hat = torch.sigmoid(out["recon_logits"])  # type: ignore[index]
                            recon_mean = F.mse_loss(x_hat, x, reduction="mean")
                            recon_per_image = recon_mean * float(x[0].numel())
                        kl = kl_diag_gaussian(out["mu"], out["logvar"])  # type: ignore[index]
                        beta = kl_sched.beta(global_step)
                        sae_reg = torch.zeros((), device=device)
                        if model_type == "combined_ae" and bool(getattr(model, "sae_enabled", True)):
                            z = out["sae_z"]  # type: ignore[index]
                            active = (z.abs() > 0)
                            l0_t = active.float().sum(dim=1).mean()
                            t_sum_l0 += float(l0_t.item()) * int(x.shape[0])
                            if t_alive_mask is not None:
                                t_alive_mask |= active.any(dim=0)
                            if combined_sae_type == "mlp_sae":
                                codes_l1 = z.abs().sum(dim=1).mean()
                                sae_reg = float(sae_l1_penalty) * codes_l1
                        loss = recon_per_image + (float(beta) * kl) + sae_reg
                    else:
                        x_flat = x.view(x.shape[0], -1)
                        z_pre, z, x_hat_flat = model(x_flat)  # type: ignore[misc]
                        if recon_loss == "bce":
                            recon_mean = F.binary_cross_entropy_with_logits(x_hat_flat, x_flat, reduction="mean")
                        else:
                            recon_mean = F.mse_loss(x_hat_flat, x_flat, reduction="mean")
                        recon_per_image = recon_mean * float(x_flat.shape[1])

                        active = (z.abs() > 0)
                        l0 = active.float().sum(dim=1).mean()
                        t_sum_l0 += float(l0.item()) * int(x.shape[0])
                        if t_alive_mask is not None:
                            t_alive_mask |= active.any(dim=0)

                        if model_type == "mlp_sae":
                            codes_l1 = z.abs().sum(dim=1).mean()
                            loss = recon_mean + (float(sae_l1_penalty) * codes_l1)
                        else:
                            loss = recon_mean

                        kl = torch.zeros((), device=device)
                        beta = 0.0
                    bs = int(x.shape[0])
                    t_seen += bs
                    t_sum_recon += float(recon_mean.item()) * bs
                    t_sum_recon_per_image += float(recon_per_image) * bs
                    t_sum_kl += float(kl.item()) * bs
                    t_sum_loss += float(loss.item()) * bs

            msg = (
                f"epoch {epoch:03d} | "
                f"train loss {sum_loss / max(1, n_seen):.4f} recon {sum_recon / max(1, n_seen):.4f} kl {sum_kl / max(1, n_seen):.4f} | "
                f"test loss {t_sum_loss / max(1, t_seen):.4f} recon {t_sum_recon / max(1, t_seen):.4f} kl {t_sum_kl / max(1, t_seen):.4f}"
            )
            print(msg, flush=True)

            if wandb_run is not None:
                t_alive_count = float("nan")
                t_alive_pct = float("nan")
                t_dead_count = float("nan")
                t_dead_pct = float("nan")
                if t_alive_mask is not None:
                    total = float(int(densae_n_sparse if model_type == "densae" else sae_n_concepts))
                    t_alive_count = float(t_alive_mask.sum().item())
                    t_alive_pct = float(t_alive_count / max(1.0, total))
                    t_dead_count = float(total - t_alive_count)
                    t_dead_pct = float(1.0 - t_alive_pct)
                eval_payload = {
                        "epoch": int(epoch),
                        "step": int(global_step),
                        "global_step": int(global_step),
                        "test/loss": float(t_sum_loss / max(1, t_seen)),
                        "test/recon": float(t_sum_recon / max(1, t_seen)),
                        "test/recon_per_image": float(t_sum_recon_per_image / max(1, t_seen)),
                        "test/kl": float(t_sum_kl / max(1, t_seen)),
                        "test/kl_beta": float(kl_sched.beta(global_step)),
                        "test/avg_l0": float(t_sum_l0 / max(1, t_seen))
                        if model_type in ("mlp_sae", "mlp_topk_sae", "mlp_mp_sae", "combined_ae", "densae")
                        else float("nan"),
                        "test/alive_features": float(t_alive_count),
                        "test/alive_features_pct": float(t_alive_pct),
                        "test/dead_features": float(t_dead_pct),
                        "test/dead_features_count": float(t_dead_count),
                }
                results.append(eval_payload)
                _write_json(results_path, results)
                wandb_run.log(eval_payload, step=int(global_step))
                _log_recon_images(
                    wandb_run=wandb_run,
                    model=model,
                    loader=train_loader,
                    device=device,
                    global_step=global_step,
                    tag="train_recon",
                    log_prior_samples=False,
                    recon_loss=recon_loss,
                    image_shape=image_shape,
                    results=results,
                )
                _log_recon_images(
                    wandb_run=wandb_run,
                    model=model,
                    loader=test_loader,
                    device=device,
                    global_step=global_step,
                    tag="test_recon",
                    log_prior_samples=True,
                    recon_loss=recon_loss,
                    image_shape=image_shape,
                    results=results,
                )
                _write_json(results_path, results)

        # --- checkpointing ---
        # Always update latest.pt every epoch (overwrite).
        _save_checkpoint(run_dir=run_dir, model=model, optimizer=opt, epoch=int(epoch), global_step=int(global_step), tag=None)
        # Also save explicit checkpoints at:
        # - epoch 1
        # - every `checkpoint_interval` epochs
        interval = max(1, int(checkpoint_interval))
        if int(epoch) == 1 or (int(epoch) % interval == 0):
            _save_checkpoint(run_dir=run_dir, model=model, optimizer=opt, epoch=int(epoch), global_step=int(global_step), tag="ckpt")

    if wandb_run is not None:
        wandb_run.finish()
    # Final write of results.
    _write_json(results_path, results)


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="VAE/SAE reconstruction on loops/lowfreq/lowbit/MNIST/combined.")
    p.add_argument(
        "--dataset",
        type=str,
        default="combined",
        choices=["loops", "lowfreq_tag", "lowbit_tag", "mnist", "combined"],
    )
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument(
        "--combined_components",
        type=str,
        nargs="+",
        default=["mnist", "lowbit_tag"],
        help='For dataset=combined. Tags from data.py: "mnist", "noise_tags", "lowfreq_tag", "lowbit_tag", "loops".',
    )
    p.add_argument(
        "--combined_space_separation",
        action="store_true",
        help="For dataset=combined. If set, combine images by concatenating side-by-side instead of pixelwise addition.",
    )
    p.add_argument(
        "--combined_tag_weight",
        type=float,
        default=1.0,
        help="For dataset=combined with --combined_space_separation. Multiply loss weight on non-MNIST slices (all pixels).",
    )
    p.add_argument(
        "--combined_tag_pos_weight",
        type=float,
        default=1.0,
        help=(
            "For dataset=combined with --combined_space_separation and BCE loss. Multiply positive-class weight "
            "on non-MNIST slices (helps very sparse lowbit tags)."
        ),
    )
    p.add_argument(
        "--combined_partition_experts",
        action="store_true",
        help=(
            "Only for --model combined_ae + dataset=combined + --combined_space_separation. "
            "Train VAE recon only on the MNIST slice and SAE recon only on non-MNIST slices."
        ),
    )
    p.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help=(
            "Global BCE pos_weight applied to ALL pixels (no region checks). "
            "Only used when --recon_loss bce. Multiplies any per-region pos weights."
        ),
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)

    p.add_argument(
        "--model",
        type=str,
        default="mlp_vae",
        choices=["mlp_vae", "conv_vae", "mlp_sae", "mlp_topk_sae", "mlp_mp_sae", "combined_ae", "densae"],
    )
    p.add_argument("--z_dim", type=int, default=16)
    p.add_argument("--mlp_hidden_dim", type=int, default=1024)
    p.add_argument("--mlp_layers", type=int, default=3)
    p.add_argument("--hidden_channels", type=int, default=32)
    p.add_argument("--recon_loss", type=str, default="bce", choices=["bce", "mse"])

    # MNIST augmentations (enabled by default for MNIST / MNIST-in-combined).
    p.add_argument("--disable_mnist_augment", action="store_true")
    p.add_argument(
        "--mnist_brightness_clip_min",
        type=float,
        default=0.0,
        help="(Deprecated / no-op) Previously enabled random brightness ceiling. Kept for backward compatibility.",
    )
    p.add_argument(
        "--mnist_max_brightness",
        type=float,
        default=1,
        help="Clamp MNIST pixels to <= this value in [0,1] space (deterministic). 1.0 disables.",
    )
    p.add_argument("--mnist_jitter_crop", type=int, default=5)
    p.add_argument("--mnist_rotation_degrees", type=float, default=10.0)
    p.add_argument("--mnist_gaussian_std", type=float, default=0.0)
    p.add_argument("--mnist_augment_apply_to_test", action="store_true")

    # SAE / TopK-SAE knobs (Overcomplete)
    p.add_argument("--sae_n_concepts", type=int, default=20_000)
    p.add_argument("--sae_l1_penalty", type=float, default=1e-4)
    p.add_argument("--topk_k", type=int, default=10)
    p.add_argument(
        "--mp_k",
        type=int,
        default=10,
        help="Only used when --model mlp_mp_sae. Number of matching pursuit iterations (k).",
    )
    p.add_argument(
        "--mp_dropout",
        type=float,
        default=None,
        help="Only used when --model mlp_mp_sae. Optional dictionary dropout probability (e.g. 0.1).",
    )
    p.add_argument("--sae_encoder_hidden_dim", type=int, default=1024)
    p.add_argument("--sae_encoder_layers", type=int, default=3)
    p.add_argument(
        "--combined_sae_type",
        type=str,
        default="mlp_topk_sae",
        choices=["mlp_sae", "mlp_topk_sae"],
        help="Only used when --model combined_ae. Select which SAE variant to combine with the MLP VAE.",
    )
    p.add_argument(
        "--dead_sae_epochs",
        type=int,
        default=0,
        help="Only used when --model combined_ae. Keep SAE disabled/frozen for this many initial epochs.",
    )

    # DenSaE (dense+sparse unrolled prox-grad) knobs
    p.add_argument("--densae_n_dense", type=int, default=256, help="Number of dense dictionary atoms (A).")
    p.add_argument("--densae_n_sparse", type=int, default=2048, help="Number of sparse dictionary atoms (B).")
    p.add_argument("--densae_iters", type=int, default=10, help="Number of unrolled iterations (T).")
    p.add_argument("--densae_step_x", type=float, default=1e-2, help="Step size for dense code updates.")
    p.add_argument("--densae_step_u", type=float, default=1e-2, help="Step size for sparse code updates.")
    p.add_argument("--densae_lambda_x", type=float, default=0.0, help="Ridge strength on dense codes.")
    p.add_argument("--densae_lambda_u", type=float, default=1e-2, help="L1 strength on sparse codes (via shrinkage).")
    p.add_argument("--densae_no_fista", action="store_true", help="Disable FISTA acceleration for DenSaE.")
    p.add_argument(
        "--densae_one_sided_u",
        action="store_true",
        help="Use one-sided (ReLU) shrinkage for sparse codes (default: twosided).",
    )

    p.add_argument("--kl_beta_max", type=float, default=1)
    p.add_argument("--kl_warmup_steps", type=int, default=0)
    p.add_argument("--kl_ramp_steps", type=int, default=10_000)

    p.add_argument("--log_every_steps", type=int, default=200)
    p.add_argument("--eval_every_epochs", type=int, default=1)

    p.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="dense-sparse-extractor-recon")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save an extra checkpoint every N epochs (in addition to latest.pt each epoch).",
    )
    p.add_argument(
        "--checkpoint_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "checkpoints"),
        help="Root directory for checkpoints; run saved under <root>/<wandb_project>/<wandb_run_name>/",
    )

    # Lowfreq dataset size knob
    p.add_argument("--lowfreq_images_per_class", type=int, default=500)
    p.add_argument(
        "--lowfreq_classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of digit classes for lowfreq tags (e.g. --lowfreq_classes 0 1 2). Default: all 0-9.",
    )
    p.add_argument(
        "--lowfreq_augment",
        action="store_true",
        help="Enable lowfreq-tag augmentations (default: off).",
    )
    p.add_argument("--lowfreq_augment_jitter_max", type=int, default=2)
    p.add_argument("--lowfreq_augment_gaussian_std", type=float, default=0.0)
    p.add_argument("--lowfreq_augment_apply_to_test", action="store_true")
    p.add_argument(
        "--lowfreq_boost",
        type=float,
        default=0.0,
        help="Low-frequency boost exponent for lowfreq tags. Set to 0.0 to disable.",
    )

    # Lowbit dataset knobs (mirrors lowfreq, but uses sparse Bernoulli bits)
    p.add_argument("--lowbit_images_per_class", type=int, default=500)
    p.add_argument(
        "--lowbit_classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of digit classes for lowbit tags (e.g. --lowbit_classes 0 1 2). Default: all 0-9.",
    )
    p.add_argument("--lowbit_p_on", type=float, default=0.03, help="Pixel-on probability for lowbit tags (default 0.05).")
    p.add_argument(
        "--lowbit_on_min_brightness",
        type=float,
        default=1.0,
        help="When a lowbit pixel is on, sample brightness ~ Uniform(on_min_brightness, 1.0). Set to 1.0 for binary 0/1.",
    )
    p.add_argument("--lowbit_augment", action="store_true", help="Enable lowbit-tag augmentations (default: off).")
    p.add_argument("--lowbit_augment_jitter_max", type=int, default=2)
    p.add_argument("--lowbit_augment_gaussian_std", type=float, default=0.0)
    p.add_argument("--lowbit_augment_apply_to_test", action="store_true")

    # Loop dataset knobs
    p.add_argument("--loop_train_length", type=int, default=200_000)
    p.add_argument("--loop_test_length", type=int, default=10_000)
    p.add_argument("--loop_seed", type=int, default=0)

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    wandb_enabled = not bool(args.disable_wandb)
    # Ensure we always have a run name (used for checkpoint foldering).
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.dataset}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    device = _resolve_device(str(args.device))

    # Avoid tokenizers/threading contention surprises.
    torch.set_num_threads(max(1, int(os.cpu_count() or 2) // 2))

    loop_cfg = LoopConfig(
        train_length=int(args.loop_train_length),
        test_length=int(args.loop_test_length),
        seed=int(args.loop_seed),
    )

    kl_sched = KLBetaSchedule(
        beta_max=float(args.kl_beta_max),
        warmup_steps=int(args.kl_warmup_steps),
        ramp_steps=int(args.kl_ramp_steps),
    )

    train(
        dataset=str(args.dataset),  # type: ignore[arg-type]
        data_dir=str(args.data_dir),
        combined_components=list(args.combined_components),
        combined_space_separation=bool(args.combined_space_separation),
        combined_tag_weight=float(args.combined_tag_weight),
        combined_tag_pos_weight=float(args.combined_tag_pos_weight),
        combined_partition_experts=bool(args.combined_partition_experts),
        pos_weight=float(args.pos_weight),
        seed=int(args.seed),
        device=device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        model_type=str(args.model),  # type: ignore[arg-type]
        z_dim=int(args.z_dim),
        hidden_channels=int(args.hidden_channels),
        mlp_hidden_dim=int(args.mlp_hidden_dim),
        mlp_layers=int(args.mlp_layers),
        sae_n_concepts=int(args.sae_n_concepts),
        sae_l1_penalty=float(args.sae_l1_penalty),
        topk_k=int(args.topk_k),
        mp_k=int(args.mp_k),
        mp_dropout=float(args.mp_dropout) if args.mp_dropout is not None else None,
        sae_encoder_hidden_dim=int(args.sae_encoder_hidden_dim),
        sae_encoder_layers=int(args.sae_encoder_layers),
        combined_sae_type=str(args.combined_sae_type),  # type: ignore[arg-type]
        dead_sae_epochs=int(args.dead_sae_epochs),
        densae_n_dense=int(args.densae_n_dense),
        densae_n_sparse=int(args.densae_n_sparse),
        densae_iters=int(args.densae_iters),
        densae_step_x=float(args.densae_step_x),
        densae_step_u=float(args.densae_step_u),
        densae_lambda_x=float(args.densae_lambda_x),
        densae_lambda_u=float(args.densae_lambda_u),
        densae_fista=not bool(args.densae_no_fista),
        densae_twosided_u=not bool(args.densae_one_sided_u),
        mnist_augment=not bool(args.disable_mnist_augment),
        mnist_brightness_clip_min=float(args.mnist_brightness_clip_min),
        mnist_max_brightness=float(args.mnist_max_brightness),
        mnist_jitter_crop=int(args.mnist_jitter_crop),
        mnist_rotation_degrees=float(args.mnist_rotation_degrees),
        mnist_gaussian_std=float(args.mnist_gaussian_std),
        mnist_augment_apply_to_test=bool(args.mnist_augment_apply_to_test),
        recon_loss=str(args.recon_loss),  # type: ignore[arg-type]
        kl_sched=kl_sched,
        log_every_steps=int(args.log_every_steps),
        eval_every_epochs=int(args.eval_every_epochs),
        wandb_enabled=bool(wandb_enabled),
        wandb_project=str(args.wandb_project),
        wandb_run_name=args.wandb_run_name,
        lowfreq_images_per_class=int(args.lowfreq_images_per_class),
        lowfreq_classes=list(args.lowfreq_classes) if args.lowfreq_classes is not None else None,
        lowfreq_augment=bool(args.lowfreq_augment),
        lowfreq_augment_jitter_max=int(args.lowfreq_augment_jitter_max),
        lowfreq_augment_gaussian_std=float(args.lowfreq_augment_gaussian_std),
        lowfreq_augment_apply_to_test=bool(args.lowfreq_augment_apply_to_test),
        lowfreq_boost=float(args.lowfreq_boost),
        lowbit_images_per_class=int(args.lowbit_images_per_class),
        lowbit_classes=list(args.lowbit_classes) if args.lowbit_classes is not None else None,
        lowbit_augment=bool(args.lowbit_augment),
        lowbit_augment_jitter_max=int(args.lowbit_augment_jitter_max),
        lowbit_augment_gaussian_std=float(args.lowbit_augment_gaussian_std),
        lowbit_augment_apply_to_test=bool(args.lowbit_augment_apply_to_test),
        lowbit_p_on=float(args.lowbit_p_on),
        lowbit_on_min_brightness=float(args.lowbit_on_min_brightness),
        loop_cfg=loop_cfg,
        checkpoint_root=str(args.checkpoint_root),
        checkpoint_interval=int(args.checkpoint_interval),
    )


if __name__ == "__main__":
    main()

