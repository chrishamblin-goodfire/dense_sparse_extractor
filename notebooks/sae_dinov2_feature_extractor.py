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
# - You have a sibling checkout at `../dinov2` (relative to this repo root).
# - You installed this repo requirements (see `requirements.txt`).

# %%
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms

from overcomplete.sae import TopKSAE, train_sae


def get_projects_dir() -> Path:
    # If you run this notebook from the repo root, this resolves to ../
    cwd = Path.cwd().resolve()
    if cwd.name == "dense_sparse_extractor":
        return cwd.parent
    # Fallback: walk up until we find this repo root, then take its parent.
    for p in [cwd, *cwd.parents]:
        if (p / "dense_sparse_extractor").is_dir() and (p / "dinov2").is_dir():
            return p
        if (p / "pyproject.toml").is_file() and (p / "dense_sparse_extractor").is_dir():
            return p.parent
    raise RuntimeError("Could not infer projects dir. Run from the repo root.")


device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
# Load DINOv2 from a local checkout via torch.hub
projects_dir = get_projects_dir()
dinov2_dir = projects_dir / "dinov2"
if not dinov2_dir.is_dir():
    raise FileNotFoundError(f"Expected dinov2 checkout at: {dinov2_dir}")


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
