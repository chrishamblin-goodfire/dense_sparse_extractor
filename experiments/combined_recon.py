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
from dataclasses import asdict, dataclass
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
        LowFreqTagConfig,
        NoiseTagConfig,
        make_combined_datasets,
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
        LowFreqTagConfig,
        NoiseTagConfig,
        make_combined_datasets,
        make_lowfreq_tag_datasets,
        make_loop_datasets,
        make_mnist_datasets,
        make_noise_tag_datasets,
    )


DatasetType = Literal["loops", "lowfreq_tag", "mnist", "combined"]
ModelType = Literal["mlp_vae", "conv_vae", "mlp_sae", "mlp_topk_sae"]


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
        hidden_dim: int = 1024,
        n_layers: int = 3,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.input_dim = int(input_dim)
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
        return logits_flat.view(z.shape[0], 1, 28, 28)

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
    lowfreq_images_per_class: int,
    combined_components: Sequence[str],
) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)

    if dataset == "loops":
        train_base, test_base = make_loop_datasets(cfg=loop_cfg)
        train_ds = XOnlyDataset(train_base)
        test_ds = XOnlyDataset(test_base)
    elif dataset == "mnist":
        train_base, test_base = make_mnist_datasets(
            data_dir=data_dir,
            normalize=False,  # recon in [0,1]
            label_format="int",
            download=True,
            augment_cfg=None,
        )
        train_ds = XOnlyDataset(train_base)
        test_ds = XOnlyDataset(test_base)
    elif dataset == "lowfreq_tag":
        lf_cfg = LowFreqTagConfig(
            images_per_class=int(lowfreq_images_per_class),
            seed=int(seed),
            normalize_like_mnist=False,  # recon in [0,1]
            cache_images=True,
            label_format="onehot",
        )
        train_base, test_base = make_lowfreq_tag_datasets(data_dir=data_dir, cfg=lf_cfg)
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
        lowfreq_cfg = LowFreqTagConfig(
            images_per_class=int(lowfreq_images_per_class),
            seed=int(seed),
            normalize_like_mnist=False,
            cache_images=True,
            label_format="onehot",
        )
        train_base, test_base = make_combined_datasets(
            data_dir=data_dir,
            noise_cfg=noise_cfg,
            lowfreq_cfg=lowfreq_cfg,
            loop_cfg=loop_cfg,
            mnist_augment_cfg=None,
            mnist_normalize=False,
            download=True,
            seed=int(seed),
            length=None,
            clip=(0.0, 1.0),
            datasets=list(combined_components),
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
    n_images: int = 64,
    tag: str = "recon",
) -> None:
    import wandb  # type: ignore

    model.eval()
    batch = next(iter(loader))
    x, _ = batch
    x = x.to(device)[: int(n_images)]
    out = model(x)

    # Support both:
    # - VAE-style dict with "recon_logits"
    # - SAE-style tuple (z_pre, z, x_hat_flat)
    if isinstance(out, dict) and "recon_logits" in out:
        x_hat = torch.sigmoid(out["recon_logits"])
    elif isinstance(out, (tuple, list)) and len(out) == 3:
        x_hat_flat = out[2]
        if not torch.is_tensor(x_hat_flat):
            raise ValueError("Unexpected SAE output type")
        # For SAE, decoder output is unconstrained; interpret it as logits and squash.
        x_hat = torch.sigmoid(x_hat_flat.view(x_hat_flat.shape[0], 1, 28, 28))
    else:
        raise ValueError("Unsupported model output format for recon logging")

    # Make a 2-row grid: originals on top, recon below.
    grid = make_grid(torch.cat([x, x_hat], dim=0), nrow=int(math.sqrt(n_images)), pad_value=1.0)
    wandb_run.log({f"images/{tag}": wandb.Image(grid), "step": int(global_step), "global_step": int(global_step)}, step=int(global_step))

    # For VAEs, also log a few random prior samples (SAEs don't have a meaningful prior).
    if hasattr(model, "decode_logits") and hasattr(model, "z_dim"):
        z_dim = int(getattr(model, "z_dim"))
        z = torch.randn(int(n_images), z_dim, device=device)
        decode_logits = getattr(model, "decode_logits")
        samples = torch.sigmoid(decode_logits(z))
        grid_s = make_grid(samples, nrow=int(math.sqrt(n_images)), pad_value=1.0)
        wandb_run.log(
            {f"images/{tag}_prior_samples": wandb.Image(grid_s), "step": int(global_step), "global_step": int(global_step)},
            step=int(global_step),
        )


def train(
    *,
    dataset: DatasetType,
    data_dir: str | Path,
    combined_components: Sequence[str],
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
    sae_encoder_hidden_dim: int,
    sae_encoder_layers: int,
    recon_loss: Literal["bce", "mse"],
    kl_sched: KLBetaSchedule,
    log_every_steps: int,
    eval_every_epochs: int,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_run_name: str | None,
    lowfreq_images_per_class: int,
    loop_cfg: LoopConfig,
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
        lowfreq_images_per_class=lowfreq_images_per_class,
        combined_components=combined_components,
    )

    if model_type == "mlp_vae":
        model: nn.Module = MLPVAE(
            z_dim=int(z_dim),
            hidden_dim=int(mlp_hidden_dim),
            n_layers=int(mlp_layers),
        ).to(device)
    elif model_type == "conv_vae":
        model = ConvVAE(z_dim=int(z_dim), hidden_channels=int(hidden_channels)).to(device)
    elif model_type in ("mlp_sae", "mlp_topk_sae"):
        oc = _maybe_import_overcomplete()
        SAE = oc.sae.SAE  # type: ignore[attr-defined]
        TopKSAE = oc.sae.TopKSAE  # type: ignore[attr-defined]
        MLPEncoder = oc.sae.MLPEncoder  # type: ignore[attr-defined]

        enc = MLPEncoder(
            28 * 28,
            int(sae_n_concepts),
            hidden_dim=int(sae_encoder_hidden_dim),
            nb_blocks=int(sae_encoder_layers),
            residual=False,
            device=str(device),
        )
        if model_type == "mlp_sae":
            model = SAE(28 * 28, int(sae_n_concepts), encoder_module=enc, device=str(device))
        else:
            model = TopKSAE(28 * 28, int(sae_n_concepts), top_k=int(topk_k), encoder_module=enc, device=str(device))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    wandb_run = None
    if bool(wandb_enabled):
        try:
            import wandb  # type: ignore
        except Exception as e:
            raise RuntimeError("wandb_enabled=True but wandb is not installed.") from e

        wandb_run = wandb.init(
            project=str(wandb_project),
            name=str(wandb_run_name) if wandb_run_name else None,
            config={
                "dataset": str(dataset),
                "combined_components": list(combined_components),
                "seed": int(seed),
                "device": str(device),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
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
                "sae_encoder_hidden_dim": int(sae_encoder_hidden_dim),
                "sae_encoder_layers": int(sae_encoder_layers),
                "recon_loss": str(recon_loss),
                "kl_schedule": asdict(kl_sched),
                "lowfreq_images_per_class": int(lowfreq_images_per_class),
                "loop_cfg": asdict(loop_cfg),
            },
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    global_step = 0
    for epoch in range(1, int(epochs) + 1):
        # Ensure epoch-dependent datasets update.
        if hasattr(train_loader.dataset, "set_epoch"):
            try:
                train_loader.dataset.set_epoch(int(epoch))  # type: ignore[attr-defined]
            except Exception:
                pass

        model.train()
        sum_recon = 0.0
        sum_kl = 0.0
        sum_loss = 0.0
        sum_l0 = 0.0
        n_seen = 0
        dead_ratio = float("nan")

        dead_tracker = None
        if model_type in ("mlp_sae", "mlp_topk_sae"):
            oc = _maybe_import_overcomplete()
            DeadCodeTracker = oc.sae.trackers.DeadCodeTracker  # type: ignore[attr-defined]
            dead_tracker = DeadCodeTracker(int(sae_n_concepts), str(device))

        for step_in_epoch, (x, _y) in enumerate(train_loader, start=1):
            x = x.to(device)
            if model_type in ("mlp_vae", "conv_vae"):
                out = model(x)
            else:
                x_flat = x.view(x.shape[0], -1)
                z_pre, z, x_hat_flat = model(x_flat)  # type: ignore[misc]
                out = (z_pre, z, x_hat_flat)

            if model_type in ("mlp_vae", "conv_vae"):
                if recon_loss == "bce":
                    # Average per-batch (sum over pixels, mean over batch)
                    recon = F.binary_cross_entropy_with_logits(out["recon_logits"], x, reduction="sum") / x.shape[0]  # type: ignore[index]
                elif recon_loss == "mse":
                    x_hat = torch.sigmoid(out["recon_logits"])  # type: ignore[index]
                    recon = F.mse_loss(x_hat, x, reduction="mean")
                else:
                    raise ValueError(f"Unknown recon_loss: {recon_loss}")

                kl = kl_diag_gaussian(out["mu"], out["logvar"])  # type: ignore[index]
                beta = kl_sched.beta(global_step)
                loss = recon + (float(beta) * kl)
                l0 = float("nan")
                dead_ratio = float("nan")
            else:
                z_pre, z, x_hat_flat = out  # type: ignore[misc]
                if recon_loss == "bce":
                    recon = F.binary_cross_entropy_with_logits(x_hat_flat, x_flat, reduction="mean")
                else:
                    recon = F.mse_loss(x_hat_flat, x_flat, reduction="mean")

                l0 = (z > 0).float().sum(dim=1).mean()
                sum_l0 += float(l0.item()) * int(x.shape[0])
                if dead_tracker is not None:
                    dead_tracker.update(z)

                # SAE loss: add optional L1 penalty for vanilla SAE.
                if model_type == "mlp_sae":
                    codes_l1 = z.abs().sum(dim=1).mean()
                    loss = recon + (float(sae_l1_penalty) * codes_l1)
                else:
                    loss = recon

                kl = torch.zeros((), device=device)
                beta = 0.0

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = int(x.shape[0])
            n_seen += bs
            sum_recon += float(recon.item()) * bs
            sum_kl += float(kl.item()) * bs
            sum_loss += float(loss.item()) * bs

            global_step += 1

            if wandb_run is not None and int(log_every_steps) > 0 and (global_step % int(log_every_steps) == 0):
                if dead_tracker is not None:
                    dead_ratio = float(dead_tracker.get_dead_ratio())
                wandb_run.log(
                    {
                        "epoch": int(epoch),
                        "step": int(global_step),
                        "global_step": int(global_step),
                        "train/loss": float(sum_loss / max(1, n_seen)),
                        "train/recon": float(sum_recon / max(1, n_seen)),
                        "train/kl": float(sum_kl / max(1, n_seen)),
                        "train/kl_beta": float(beta),
                        "train/avg_l0": float(sum_l0 / max(1, n_seen)) if model_type in ("mlp_sae", "mlp_topk_sae") else float("nan"),
                        "train/dead_features": float(dead_ratio) if model_type in ("mlp_sae", "mlp_topk_sae") else float("nan"),
                    },
                    step=int(global_step),
                )

        # Evaluate
        if int(eval_every_epochs) > 0 and (epoch % int(eval_every_epochs) == 0):
            model.eval()
            t_sum_recon = 0.0
            t_sum_kl = 0.0
            t_sum_loss = 0.0
            t_sum_l0 = 0.0
            t_seen = 0
            t_dead_ratio = float("nan")
            t_dead_tracker = None
            if model_type in ("mlp_sae", "mlp_topk_sae"):
                oc = _maybe_import_overcomplete()
                DeadCodeTracker = oc.sae.trackers.DeadCodeTracker  # type: ignore[attr-defined]
                t_dead_tracker = DeadCodeTracker(int(sae_n_concepts), str(device))
            with torch.no_grad():
                for x, _y in test_loader:
                    x = x.to(device)
                    if model_type in ("mlp_vae", "conv_vae"):
                        out = model(x)
                        if recon_loss == "bce":
                            recon = F.binary_cross_entropy_with_logits(out["recon_logits"], x, reduction="sum") / x.shape[0]  # type: ignore[index]
                        else:
                            x_hat = torch.sigmoid(out["recon_logits"])  # type: ignore[index]
                            recon = F.mse_loss(x_hat, x, reduction="mean")
                        kl = kl_diag_gaussian(out["mu"], out["logvar"])  # type: ignore[index]
                        beta = kl_sched.beta(global_step)
                        loss = recon + (float(beta) * kl)
                    else:
                        x_flat = x.view(x.shape[0], -1)
                        z_pre, z, x_hat_flat = model(x_flat)  # type: ignore[misc]
                        if recon_loss == "bce":
                            recon = F.binary_cross_entropy_with_logits(x_hat_flat, x_flat, reduction="mean")
                        else:
                            recon = F.mse_loss(x_hat_flat, x_flat, reduction="mean")

                        l0 = (z > 0).float().sum(dim=1).mean()
                        t_sum_l0 += float(l0.item()) * int(x.shape[0])
                        if t_dead_tracker is not None:
                            t_dead_tracker.update(z)

                        if model_type == "mlp_sae":
                            codes_l1 = z.abs().sum(dim=1).mean()
                            loss = recon + (float(sae_l1_penalty) * codes_l1)
                        else:
                            loss = recon

                        kl = torch.zeros((), device=device)
                        beta = 0.0
                    bs = int(x.shape[0])
                    t_seen += bs
                    t_sum_recon += float(recon.item()) * bs
                    t_sum_kl += float(kl.item()) * bs
                    t_sum_loss += float(loss.item()) * bs

            msg = (
                f"epoch {epoch:03d} | "
                f"train loss {sum_loss / max(1, n_seen):.4f} recon {sum_recon / max(1, n_seen):.4f} kl {sum_kl / max(1, n_seen):.4f} | "
                f"test loss {t_sum_loss / max(1, t_seen):.4f} recon {t_sum_recon / max(1, t_seen):.4f} kl {t_sum_kl / max(1, t_seen):.4f}"
            )
            print(msg, flush=True)

            if wandb_run is not None:
                if t_dead_tracker is not None:
                    t_dead_ratio = float(t_dead_tracker.get_dead_ratio())
                wandb_run.log(
                    {
                        "epoch": int(epoch),
                        "step": int(global_step),
                        "global_step": int(global_step),
                        "test/loss": float(t_sum_loss / max(1, t_seen)),
                        "test/recon": float(t_sum_recon / max(1, t_seen)),
                        "test/kl": float(t_sum_kl / max(1, t_seen)),
                        "test/kl_beta": float(kl_sched.beta(global_step)),
                        "test/avg_l0": float(t_sum_l0 / max(1, t_seen)) if model_type in ("mlp_sae", "mlp_topk_sae") else float("nan"),
                        "test/dead_features": float(t_dead_ratio) if model_type in ("mlp_sae", "mlp_topk_sae") else float("nan"),
                    },
                    step=int(global_step),
                )
                _log_recon_images(wandb_run=wandb_run, model=model, loader=test_loader, device=device, global_step=global_step)

    if wandb_run is not None:
        wandb_run.finish()


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="VAE reconstruction on loops/lowfreq/MNIST/combined.")
    p.add_argument("--dataset", type=str, default="combined", choices=["loops", "lowfreq_tag", "mnist", "combined"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument(
        "--combined_components",
        type=str,
        nargs="+",
        default=["loops", "lowfreq_tag"],
        help='For dataset=combined. Tags from data.py: "mnist", "noise_tags", "lowfreq_tag", "loops".',
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)

    p.add_argument(
        "--model",
        type=str,
        default="mlp_vae",
        choices=["mlp_vae", "conv_vae", "mlp_sae", "mlp_topk_sae"],
    )
    p.add_argument("--z_dim", type=int, default=16)
    p.add_argument("--mlp_hidden_dim", type=int, default=1024)
    p.add_argument("--mlp_layers", type=int, default=3)
    p.add_argument("--hidden_channels", type=int, default=32)
    p.add_argument("--recon_loss", type=str, default="bce", choices=["bce", "mse"])

    # SAE / TopK-SAE knobs (Overcomplete)
    p.add_argument("--sae_n_concepts", type=int, default=4096)
    p.add_argument("--sae_l1_penalty", type=float, default=1e-3)
    p.add_argument("--topk_k", type=int, default=20)
    p.add_argument("--sae_encoder_hidden_dim", type=int, default=1024)
    p.add_argument("--sae_encoder_layers", type=int, default=3)

    p.add_argument("--kl_beta_max", type=float, default=10)
    p.add_argument("--kl_warmup_steps", type=int, default=0)
    p.add_argument("--kl_ramp_steps", type=int, default=10_000)

    p.add_argument("--log_every_steps", type=int, default=200)
    p.add_argument("--eval_every_epochs", type=int, default=1)

    p.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="dense-sparse-extractor-recon")
    p.add_argument("--wandb_run_name", type=str, default=None)

    # Lowfreq dataset size knob
    p.add_argument("--lowfreq_images_per_class", type=int, default=200)

    # Loop dataset knobs
    p.add_argument("--loop_train_length", type=int, default=200_000)
    p.add_argument("--loop_test_length", type=int, default=10_000)
    p.add_argument("--loop_seed", type=int, default=0)

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    wandb_enabled = not bool(args.disable_wandb)
    if wandb_enabled and args.wandb_run_name is None:
        from datetime import datetime
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
        sae_encoder_hidden_dim=int(args.sae_encoder_hidden_dim),
        sae_encoder_layers=int(args.sae_encoder_layers),
        recon_loss=str(args.recon_loss),  # type: ignore[arg-type]
        kl_sched=kl_sched,
        log_every_steps=int(args.log_every_steps),
        eval_every_epochs=int(args.eval_every_epochs),
        wandb_enabled=bool(wandb_enabled),
        wandb_project=str(args.wandb_project),
        wandb_run_name=args.wandb_run_name,
        lowfreq_images_per_class=int(args.lowfreq_images_per_class),
        loop_cfg=loop_cfg,
    )


if __name__ == "__main__":
    main()

