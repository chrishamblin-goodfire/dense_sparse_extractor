# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Dense Sparse Extractor
#     language: python
#     name: dense_sparse_extractor
# ---

# %% [markdown]
# ## SAE feature extractor on DINOv2 patch activations
#
# This notebook demonstrates a simple end-to-end pipeline:
#
# - Load a DINOv2 ViT backbone **from a local `dinov2/` checkout** via `torch.hub.load(..., source=\"local\")`
# - Run a small batch of images through the model and extract **patch-token activations**
# - Train a small **Top-k Sparse Autoencoder (SAE)** using `overcomplete`
# - Use the SAE codes as a sparse feature representation
#
# ### Assumptions
#
# - You have DINOv2 vendored into this repo at `dense_sparse_extractor/models/dino_v2`
#   (or `dense_sparse_extractor/models/dinov2`).
# - You installed this repo requirements (see `requirements.txt`).

# %%
from __future__ import annotations

from pathlib import Path
import subprocess

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms

from overcomplete.sae import TopKSAE, train_sae


def find_repo_root() -> Path:
    """
    Find the `dense_sparse_extractor` repo root without assuming anything outside it.
    """
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "pyproject.toml").is_file() and (p / "dense_sparse_extractor").is_dir():
            return p
    raise RuntimeError("Could not find repo root. Run from the repo root (the folder containing pyproject.toml).")


def ensure_dinov2_repo(repo_root: Path) -> Path:
    """
    Ensure DINOv2 is available inside this repo (auto-download if missing).

    Uses a single canonical location:
      - dense_sparse_extractor/models/dino_v2/
    """
    target_dir = repo_root / "dense_sparse_extractor" / "models" / "dino_v2"
    hubconf = target_dir / "hubconf.py"

    if hubconf.is_file():
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() and not hubconf.is_file():
        # Directory exists but isn't a usable DINOv2 checkout.
        raise FileNotFoundError(
            f"Found {target_dir}, but it doesn't look like a DINOv2 checkout (missing hubconf.py). "
            "Delete it or replace it with the DINOv2 repo."
        )

    print(f"[info] DINOv2 not found; cloning into: {target_dir}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/facebookresearch/dinov2.git", str(target_dir)],
            check=True,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to download DINOv2 automatically. "
            "Make sure you have `git` installed and network access, or clone it manually into "
            f"{target_dir}."
        ) from e

    if not hubconf.is_file():
        raise RuntimeError(f"DINOv2 clone completed but {hubconf} is still missing.")
    return target_dir


device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
# Load DINOv2 from a local checkout via torch.hub
repo_root = find_repo_root()
dinov2_dir = ensure_dinov2_repo(repo_root)


def load_dinov2_backbone(*, repo_dir: Path, name: str = "dinov2_vits14", device: str, pretrained: bool = True):
    """
    Load a DINOv2 backbone from a *local* checkout using torch.hub.

    If `pretrained=True` fails (e.g. no network and weights not cached), falls back to `pretrained=False`.
    """
    try:
        model = torch.hub.load(str(repo_dir), name, source="local", pretrained=pretrained)
    except Exception as e:
        if pretrained:
            print("[warn] Failed to load pretrained weights; falling back to pretrained=False.")
            print("       Error:", repr(e))
            model = torch.hub.load(str(repo_dir), name, source="local", pretrained=False)
        else:
            raise

    model.eval()
    return model.to(device)


model = load_dinov2_backbone(repo_dir=dinov2_dir, name="dinov2_vits14", device=device, pretrained=True)

# Fake images (no downloads). Normalize like ImageNet.
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

dataset = FakeData(size=32, image_size=(3, 224, 224), num_classes=10, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

batch = next(iter(loader))
images, labels = batch
images = images.to(device)
images.shape

# %%
# Extract patch-token activations from DINOv2
# forward_features returns a dict with (B, N, D) patch tokens under x_norm_patchtokens.
with torch.no_grad():
    feats = model.forward_features(images)
    patch_tokens = feats["x_norm_patchtokens"]  # (B, N, D)

B, N, D = patch_tokens.shape
activations = patch_tokens.reshape(B * N, D).contiguous()
patch_tokens.shape, activations.shape

# %%
# Train a small TopKSAE on these activations (toy demo settings)
sae = TopKSAE(input_shape=D, nb_concepts=512, top_k=16, device=device)

act_loader = DataLoader(activations, batch_size=1024, shuffle=True)
opt = torch.optim.Adam(sae.parameters(), lr=5e-4)

def mse_criterion(x, x_hat, pre_codes, codes, dictionary):
    return (x - x_hat).pow(2).mean()

logs = train_sae(sae, act_loader, mse_criterion, opt, nb_epochs=2, device=device)

# Use SAE as a sparse feature extractor
sae.eval()
with torch.no_grad():
    _, codes = sae.encode(activations)  # (B*N, nb_concepts) with top_k non-zeros

codes.shape, (codes != 0).float().sum(dim=-1).mean().item()

# %%
