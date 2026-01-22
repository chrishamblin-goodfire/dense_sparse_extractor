#!/usr/bin/env python3
# %%
"""
Train a Sparse Autoencoder (SAE) on activations from a trained MNIST model checkpoint.

This script is written with `# %%` cell markers so you can run it as a notebook
in editors like VS Code / Cursor, while also being runnable as a normal script.

Usage (notebook-style):
  - Edit the "Parameters" cell below.
  - Run cells top-to-bottom.
"""

# %%
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

# NOTE: this script assumes your environment has torch + overcomplete installed.
# If you don't have overcomplete yet:
#   pip install overcomplete
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# %%
# -----------------------
# Parameters (edit these)
# -----------------------
CHECKPOINT_PATH = "../checkpoints/dense-sparse-extractor_base_model/mlp_lowfreq/latest.pt"

# "auto" picks CUDA if available.
DEVICE = "auto"  # "auto" | "cpu" | "cuda"

# Feature extractor type:
# - "sae": Top-k Sparse Autoencoder (overcomplete)
# - "vae": Variational Autoencoder (same encode() I/O shape as SAE codes)
# - "svae": SAE + VAE in parallel; reconstructions are summed: x_hat = x_hat_sae + x_hat_vae
EXTRACTOR_TYPE = "sae"  # "sae" | "vae" | "svae"

# Module name to hook (printed by the layer options cell). If empty, a default is chosen.
LAYER_NAME = ""

# If the hooked activation is (B, T, d), include token index 0 (CLS). Default: drop it.
INCLUDE_CLS_TOKEN = False

# Printing / collection
MAX_PRINT_LAYERS = 200
# Set to an int cap (e.g. 200_000) or to None / 0 to collect **all** activations from the loader.
MAX_ACTIVATIONS: int | None = None
PROGRESS_EVERY = 50

# Data: still uses the **train** loader, but you can disable any MNIST augmentations.
DISABLE_AUGMENTATIONS = True

# SAE hyperparameters (TopK SAE)
SAE_NB_CONCEPTS = 4_000
SAE_TOP_K = 10
SAE_LR = 5e-4
SAE_BATCH_SIZE = 1024
SAE_EPOCHS = 30

# VAE hyperparameters (activation VAE)
# NOTE: to be swappable with SAE downstream, default latent dim matches SAE_NB_CONCEPTS.
VAE_LATENT_DIM = 5
VAE_HIDDEN_DIM = 128
VAE_BETA = 1.0
VAE_LR = 5e-4
VAE_BATCH_SIZE = 1024
VAE_EPOCHS = 30

# SVAE hyperparameters (SAE+VAE hybrid)
# Reconstruction is MSE on (x_hat_sae + x_hat_vae); KL is applied only to the VAE branch.
SVAE_BETA = 1.0
SVAE_LR = 5e-4
SVAE_BATCH_SIZE = 1024
SVAE_EPOCHS = 30

# Output directory for SAE checkpoint (empty => alongside model checkpoint)
OUT_DIR = ""

# Extra: train one SAE per encoder block (ViT only)
TRAIN_SAE_PER_VIT_BLOCK = True
# Which residual-stream points to train on:
# - "blocks": trains on encoder.layers.{i} outputs (post-block residual stream)
# - "blocks+final": also includes "encoder" (post-last block, pre-final LayerNorm)
# - "blocks+final+norm": also includes "norm" (post-final LayerNorm)
VIT_BLOCK_LAYER_SET = "blocks"


# %%
def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))
    return torch.load(str(ckpt_path), map_location="cpu")


def _training_config_from_dict(d: dict[str, Any]):
    # Reuse the TrainingConfig definition from the training entrypoint.
    from dense_sparse_extractor.train import TrainingConfig

    allowed = {f.name for f in fields(TrainingConfig)}
    filtered = {k: v for k, v in dict(d).items() if k in allowed}
    return TrainingConfig(**filtered)


def _build_model_from_cfg(model_cfg: dict[str, Any]) -> nn.Module:
    # Uses the same builder as training (ensures state_dict compatibility).
    from dense_sparse_extractor.train import _build_model

    return _build_model(dict(model_cfg))


def _make_loaders_from_training_cfg(*, train_cfg, device: torch.device) -> tuple[DataLoader, DataLoader]:
    from dense_sparse_extractor.train import _make_loaders

    return _make_loaders(cfg=train_cfg, device=device)


def _format_shape(x: Any) -> str:
    if torch.is_tensor(x):
        return f"Tensor{tuple(x.shape)} {str(x.dtype).replace('torch.', '')}"
    return f"{type(x).__name__}"


def _first_tensor(out: Any) -> torch.Tensor | None:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (list, tuple)):
        for x in out:
            if torch.is_tensor(x):
                return x
    if isinstance(out, dict):
        for x in out.values():
            if torch.is_tensor(x):
                return x
    return None


@torch.no_grad()
def list_layer_options(
    *,
    model: nn.Module,
    example_batch: torch.Tensor,
    device: torch.device,
    name_filter: Callable[[str, nn.Module], bool] | None = None,
    max_print: int = 200,
) -> list[tuple[str, str, str]]:
    """
    Run one forward pass and print candidate module hook points.

    Returns a list of tuples: (module_name, module_type, output_shape_str).
    """
    model.eval()

    records: list[tuple[str, str, str]] = []
    hooks: list[Any] = []

    def should_keep(name: str, m: nn.Module) -> bool:
        if name == "":
            return False
        if name_filter is not None:
            return bool(name_filter(name, m))
        # Default: keep modules that commonly produce useful activations.
        keep_types = (
            nn.Conv2d,
            nn.Linear,
            nn.LayerNorm,
            nn.BatchNorm2d,
            nn.TransformerEncoderLayer,
            nn.TransformerEncoder,
        )
        return isinstance(m, keep_types)

    def mk_hook(name: str, m: nn.Module):
        def _hook(_m, _inp, out):
            t = _first_tensor(out)
            if t is None:
                return
            # Avoid huge prints by skipping scalars / weird outputs.
            if t.ndim < 2:
                return
            records.append((name, type(m).__name__, _format_shape(t)))

        return _hook

    for name, m in model.named_modules():
        if should_keep(name, m):
            hooks.append(m.register_forward_hook(mk_hook(name, m)))

    _ = model(example_batch.to(device))

    for h in hooks:
        h.remove()

    # Deduplicate by name, keep first occurrence.
    seen: set[str] = set()
    uniq: list[tuple[str, str, str]] = []
    for name, typ, shp in sorted(records, key=lambda r: r[0]):
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, typ, shp))

    print("\n=== Layer / module hook options (name | type | output) ===")
    for i, (name, typ, shp) in enumerate(uniq[: int(max_print)]):
        print(f"{i:03d}  {name} | {typ} | {shp}")
    if len(uniq) > int(max_print):
        print(f"... ({len(uniq) - int(max_print)} more; increase --max_print_layers)")
    print("=========================================================\n")

    return uniq


def _activations_to_2d(*, act: torch.Tensor, include_cls_token: bool) -> torch.Tensor:
    """
    Convert a module activation tensor to (N, d) for SAE training.

    Supported shapes:
    - (B, d)           -> (B, d)
    - (B, T, d)        -> (B*T, d)   (optionally dropping cls token at index 0)
    - (B, d, H, W)     -> (B*H*W, d)
    """
    if act.ndim == 2:
        return act

    if act.ndim == 3:
        # (B, T, d)
        if not include_cls_token and act.shape[1] >= 2:
            act = act[:, 1:, :]
        return act.reshape(-1, act.shape[-1])

    if act.ndim == 4:
        # (B, d, H, W) -> (B*H*W, d)
        act = act.permute(0, 2, 3, 1).contiguous()
        return act.reshape(-1, act.shape[-1])

    raise ValueError(f"Unsupported activation shape for SAE: {tuple(act.shape)}")


@torch.no_grad()
def collect_activations(
    *,
    model: nn.Module,
    loader: DataLoader,
    layer_name: str,
    device: torch.device,
    max_activations: int | None,
    include_cls_token: bool,
    progress_every: int = 50,
) -> torch.Tensor:
    """
    Collect activation vectors (N, d) from `layer_name`.

    If `max_activations` is an int > 0, collects up to that many.
    If `max_activations` is None or <= 0, collects **all** activations from the loader.
    """
    # Resolve module.
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise KeyError(
            f"Unknown layer_name {layer_name!r}. "
            f"Call with --layer list (or omit --layer) to see available names."
        )
    target = modules[layer_name]

    cap: int | None
    if max_activations is None:
        cap = None
    else:
        cap = int(max_activations)
        if cap <= 0:
            cap = None

    buf: list[torch.Tensor] = []
    n_total = 0

    def _hook(_m, _inp, out):
        nonlocal n_total
        t = _first_tensor(out)
        if t is None:
            return
        t = t.detach()
        t2d = _activations_to_2d(act=t, include_cls_token=include_cls_token)
        if t2d.numel() == 0:
            return
        # Move to CPU for storage; keep float32.
        t2d = t2d.to(dtype=torch.float32, device="cpu")

        if cap is not None:
            remaining = int(cap) - int(n_total)
            if remaining <= 0:
                return
            if t2d.shape[0] > remaining:
                t2d = t2d[:remaining]
        buf.append(t2d)
        n_total += int(t2d.shape[0])

    h = target.register_forward_hook(_hook)
    try:
        model.eval()
        for step, (x, _y) in enumerate(loader, start=1):
            if cap is not None and n_total >= int(cap):
                break
            _ = model(x.to(device))
            if progress_every and step % int(progress_every) == 0:
                if cap is None:
                    print(f"Collected {n_total:,} activations...")
                else:
                    print(f"Collected {n_total:,}/{int(cap):,} activations...")
    finally:
        h.remove()

    if not buf:
        raise RuntimeError("No activations were collected. Is the layer producing tensor outputs?")
    acts = torch.cat(buf, dim=0)
    if cap is not None and acts.shape[0] > int(cap):
        acts = acts[: int(cap)]
    print(f"Final activations tensor: {tuple(acts.shape)} (on CPU)")
    return acts


# %%
class ActivationVAE(nn.Module):
    """
    Simple MLP VAE for (N, d_in) activations.

    Compatibility goal: expose `encode(x) -> (pre_codes, codes)` like `TopKSAE`,
    so downstream feature extraction can do:
        _, codes = extractor.encode(x)
    """

    def __init__(
        self,
        *,
        d_in: int,
        d_latent: int,
        hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_latent = int(d_latent)
        self.hidden_dim = int(hidden_dim)

        self.enc = nn.Sequential(
            nn.Linear(self.d_in, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(self.hidden_dim, self.d_latent)
        self.logvar = nn.Linear(self.hidden_dim, self.d_latent)

        self.dec = nn.Sequential(
            nn.Linear(self.d_latent, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.d_in),
        )

    def encode_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor, *, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        SAE-compatible: returns (pre_codes, codes).
        For VAE, we use the mean as the deterministic "code".
        """
        mu, _logvar = self.encode_stats(x)
        return mu, mu


class ActivationSVAE(nn.Module):
    """
    SAE + VAE in parallel, where reconstructions are summed:
        x_hat = x_hat_sae + x_hat_vae

    This lets the SAE branch learn a sparse "dictionary-like" component while the
    VAE branch can model a smooth residual.

    `encode(x)` returns concatenated codes: [sae_codes, vae_mu].
    """

    def __init__(self, *, sae: nn.Module, vae: ActivationVAE) -> None:
        super().__init__()
        self.sae = sae
        self.vae = vae

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Overcomplete TopKSAE: (z_pre, z, x_hat)
        sae_pre, sae_z, sae_x_hat = self.sae(x)
        vae_x_hat, mu, logvar, z = self.vae(x, sample=True)
        x_hat = sae_x_hat + vae_x_hat
        return {
            "x_hat": x_hat,
            "sae_pre": sae_pre,
            "sae_z": sae_z,
            "sae_x_hat": sae_x_hat,
            "vae_mu": mu,
            "vae_logvar": logvar,
            "vae_z": z,
            "vae_x_hat": vae_x_hat,
        }

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Deterministic codes: SAE uses z, VAE uses mu.
        sae_pre, sae_z, _sae_x_hat = self.sae(x)
        mu, _logvar = self.vae.encode_stats(x)
        pre = torch.cat([sae_pre, mu], dim=-1)
        codes = torch.cat([sae_z, mu], dim=-1)
        return pre, codes


def train_vae(
    *,
    vae: ActivationVAE,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    beta: float,
    epochs: int,
) -> dict[str, list[float]]:
    vae.train()
    opt = torch.optim.Adam(vae.parameters(), lr=float(lr))
    mse = nn.MSELoss(reduction="mean")

    logs: dict[str, list[float]] = {
        "loss/total": [],
        "loss/recon": [],
        "loss/kl": [],
        "r2": [],
    }

    for epoch in range(1, int(epochs) + 1):
        vae.train()
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        n_batches = 0

        # For epoch-level R2: accumulate SSE and SST over all elements.
        n_total = 0
        sum_x = None
        sum_x2 = None
        sse = 0.0

        for batch in loader:
            # TensorDataset yields a 1-tuple.
            x = batch[0].to(device)
            x_hat, mu, logvar, _z = vae(x, sample=True)

            recon = mse(x_hat, x)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + float(beta) * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss_sum += float(loss.detach().item())
            recon_loss_sum += float(recon.detach().item())
            kl_loss_sum += float(kl.detach().item())
            n_batches += 1

            # R2 accumulators (CPU, float64 for stability).
            x_cpu = x.detach().to(dtype=torch.float64, device="cpu")
            x_hat_cpu = x_hat.detach().to(dtype=torch.float64, device="cpu")
            n_total += int(x_cpu.shape[0])
            if sum_x is None:
                sum_x = x_cpu.sum(dim=0)
                sum_x2 = (x_cpu.pow(2)).sum(dim=0)
            else:
                sum_x += x_cpu.sum(dim=0)
                sum_x2 += (x_cpu.pow(2)).sum(dim=0)
            sse += float((x_cpu - x_hat_cpu).pow(2).sum().item())

        mean_total = total_loss_sum / max(1, n_batches)
        mean_recon = recon_loss_sum / max(1, n_batches)
        mean_kl = kl_loss_sum / max(1, n_batches)

        # Compute SST = sum(x^2) - sum(x)^2 / n across all dims.
        if sum_x is None or sum_x2 is None or n_total <= 1:
            r2 = float("nan")
        else:
            sst = float(sum_x2.sum().item() - (sum_x.pow(2).sum().item() / float(n_total)))
            r2 = float(1.0 - (sse / sst)) if sst > 0 else float("nan")

        logs["loss/total"].append(float(mean_total))
        logs["loss/recon"].append(float(mean_recon))
        logs["loss/kl"].append(float(mean_kl))
        logs["r2"].append(float(r2))

        if epoch == 1 or epoch == int(epochs) or (epoch % 10 == 0):
            print(
                f"epoch {epoch:03d}/{int(epochs)} | "
                f"loss {mean_total:.6f} (recon {mean_recon:.6f}, kl {mean_kl:.6f}) | r2 {r2:.4f}"
            )

    return logs


def train_svae(
    *,
    svae: ActivationSVAE,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    beta: float,
    epochs: int,
) -> dict[str, list[float]]:
    """
    Train SVAE end-to-end with:
      recon = MSE(x_hat_sae + x_hat_vae, x)
      kl    = KL(q(z|x) || N(0,I)) for VAE branch
      loss  = recon + beta * kl

    Also tracks "dead_features" based on SAE codes never being >0 during an epoch.
    """
    svae.train()
    opt = torch.optim.Adam(svae.parameters(), lr=float(lr))
    mse = nn.MSELoss(reduction="mean")
    eps = 1e-6

    logs: dict[str, list[float]] = {
        "loss/total": [],
        "loss/recon": [],
        "loss/kl": [],
        "r2": [],
        "dead_features": [],
    }

    for epoch in range(1, int(epochs) + 1):
        svae.train()
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        r2_sum = 0.0
        n_batches = 0

        alive = None  # bool[nb_sae_features]

        for batch in loader:
            x = batch[0].to(device)
            out = svae(x)
            x_hat = out["x_hat"]
            mu = out["vae_mu"]
            logvar = out["vae_logvar"]
            sae_z = out["sae_z"]

            recon = mse(x_hat, x)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + float(beta) * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # R2 (match Overcomplete's scalar-mean variant)
            ss_res = torch.mean((x - x_hat) ** 2)
            ss_tot = torch.mean((x - x.mean()) ** 2)
            r2 = 1.0 - (ss_res / (ss_tot + eps))

            total_loss_sum += float(loss.detach().item())
            recon_loss_sum += float(recon.detach().item())
            kl_loss_sum += float(kl.detach().item())
            r2_sum += float(r2.detach().item())
            n_batches += 1

            # Dead feature tracking (epoch-wide)
            z_alive = (sae_z.detach() > 0).any(dim=0)
            alive = z_alive if alive is None else (alive | z_alive)

        mean_total = total_loss_sum / max(1, n_batches)
        mean_recon = recon_loss_sum / max(1, n_batches)
        mean_kl = kl_loss_sum / max(1, n_batches)
        mean_r2 = r2_sum / max(1, n_batches)
        dead_ratio = float("nan") if alive is None else float(1.0 - alive.float().mean().item())

        logs["loss/total"].append(float(mean_total))
        logs["loss/recon"].append(float(mean_recon))
        logs["loss/kl"].append(float(mean_kl))
        logs["r2"].append(float(mean_r2))
        logs["dead_features"].append(float(dead_ratio))

        if epoch == 1 or epoch == int(epochs) or (epoch % 10 == 0):
            print(
                f"epoch {epoch:03d}/{int(epochs)} | "
                f"loss {mean_total:.6f} (recon {mean_recon:.6f}, kl {mean_kl:.6f}) | "
                f"r2 {mean_r2:.4f} | dead {dead_ratio*100:.1f}%"
            )

    return logs


# %%
# Notebook cell: load checkpoint, data, model; list layers; collect activations; train SAE; save.

# --- Load checkpoint ---
ckpt = _load_checkpoint(CHECKPOINT_PATH)
train_cfg_dict = dict(ckpt.get("training_config", {}))
model_cfg = dict(ckpt.get("model_config", {}))
if not train_cfg_dict or not model_cfg:
    raise ValueError("Checkpoint missing training_config/model_config. Is this from dense_sparse_extractor.train?")

device = _device_from_arg(DEVICE)
print(f"Using device for forward passes: {device}")

# --- Rebuild data loaders using the same data generating process ---
if bool(DISABLE_AUGMENTATIONS):
    # TrainingConfig is frozen, so we edit the dict before reconstructing it.
    # - For MNIST: clear the mnist augmentation config blob.
    if isinstance(train_cfg_dict.get("mnist", None), dict):
        train_cfg_dict["mnist"] = {}
    # - For combined dataset: clear any nested override at combined.mnist.
    if isinstance(train_cfg_dict.get("combined", None), dict):
        combined_cfg = dict(train_cfg_dict["combined"])
        combined_cfg.pop("mnist", None)
        train_cfg_dict["combined"] = combined_cfg

train_cfg = _training_config_from_dict(train_cfg_dict)
train_loader, test_loader = _make_loaders_from_training_cfg(train_cfg=train_cfg, device=device)
print(
    "Loaded dataset via training config:",
    f"dataset={train_cfg.dataset} batch_size={train_cfg.batch_size} data_dir={train_cfg.data_dir}",
)

# --- Rebuild model & load weights ---
model = _build_model_from_cfg(model_cfg).to(device)
missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
if missing:
    print(f"Warning: missing keys when loading model_state_dict ({len(missing)}). Example: {missing[:5]}")
if unexpected:
    print(f"Warning: unexpected keys when loading model_state_dict ({len(unexpected)}). Example: {unexpected[:5]}")
model.eval()

# --- Print layer options for this architecture ---
x0, _y0 = next(iter(train_loader))
_ = list_layer_options(
    model=model,
    example_batch=x0,
    device=device,
    max_print=int(MAX_PRINT_LAYERS),
)

# Choose default layer if not provided.
layer = str(LAYER_NAME).strip()
if not layer:
    model_type = str(model_cfg.get("model_type", "")).lower()
    if model_type == "vit":
        layer = "encoder"
    elif model_type == "convnet":
        layer = "features"
    elif model_type == "mlp":
        layer = "net"
    else:
        layer = "encoder"
    print(f"LAYER_NAME is empty. Using default for model_type={model_type!r}: {layer!r}")
else:
    print(f"Using requested layer: {layer!r}")



# %%
# --- Collect activations from training data ---
acts = collect_activations(
    model=model,
    loader=train_loader,
    layer_name=layer,
    device=device,
    max_activations=MAX_ACTIVATIONS,
    include_cls_token=bool(INCLUDE_CLS_TOKEN),
    progress_every=int(PROGRESS_EVERY),
)

# --- Train feature extractor (SAE or VAE) ---
ckpt_path = Path(CHECKPOINT_PATH)
out_dir = Path(OUT_DIR) if str(OUT_DIR).strip() else (ckpt_path.parent / "sae")
out_dir.mkdir(parents=True, exist_ok=True)

d_in = int(acts.shape[1])
extractor_type = str(EXTRACTOR_TYPE).strip().lower()
if extractor_type not in ("sae", "vae", "svae"):
    raise ValueError(f"Unknown EXTRACTOR_TYPE={EXTRACTOR_TYPE!r} (expected 'sae', 'vae', or 'svae')")

ds = TensorDataset(acts)
if extractor_type == "sae":
    # --- Train SAE using Overcomplete ---
    try:
        from overcomplete.sae import TopKSAE, train_sae
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Failed to import overcomplete. Install it with: pip install overcomplete") from e

    sae = TopKSAE(d_in, nb_concepts=int(SAE_NB_CONCEPTS), top_k=int(SAE_TOP_K), device=str(device))
    dl = DataLoader(ds, batch_size=int(SAE_BATCH_SIZE), shuffle=True, drop_last=True)
    opt = torch.optim.Adam(sae.parameters(), lr=float(SAE_LR))

    def criterion(x, x_hat, pre_codes, codes, dictionary):
        # Standard reconstruction loss; add extra regularizers here if desired.
        return (x - x_hat).pow(2).mean()

    print(
        f"Training SAE on activations: N={acts.shape[0]:,} d={acts.shape[1]} "
        f"nb_concepts={int(SAE_NB_CONCEPTS):,} top_k={int(SAE_TOP_K)}"
    )
    logs = train_sae(sae, dl, criterion, opt, nb_epochs=int(SAE_EPOCHS), device=str(device))

    extractor_path = out_dir / f"sae_{ckpt_path.stem}_layer-{layer.replace('.', '_')}.pt"
    payload = {
        # Back-compat (existing keys)
        "sae_state_dict": sae.state_dict(),
        "sae_class": type(sae).__name__,
        "sae_config": {
            "d_in": d_in,
            "nb_concepts": int(SAE_NB_CONCEPTS),
            "top_k": int(SAE_TOP_K),
        },
        # Unified keys (new)
        "extractor_type": "sae",
        "extractor_state_dict": sae.state_dict(),
        "extractor_class": type(sae).__name__,
        "extractor_config": {
            "d_in": d_in,
            "feature_dim": int(SAE_NB_CONCEPTS),
            "top_k": int(SAE_TOP_K),
        },
        "source_checkpoint": str(ckpt_path),
        "source_model_config": model_cfg,
        "source_training_config": train_cfg_dict,
        "layer_name": layer,
        "include_cls_token": bool(INCLUDE_CLS_TOKEN),
        "max_activations": MAX_ACTIVATIONS,
        "disable_augmentations": bool(DISABLE_AUGMENTATIONS),
        "logs": logs,
    }
    torch.save(payload, extractor_path)
    print(f"Saved SAE checkpoint to: {extractor_path}")

elif extractor_type == "vae":
    # --- Train VAE ---
    vae = ActivationVAE(d_in=d_in, d_latent=int(VAE_LATENT_DIM), hidden_dim=int(VAE_HIDDEN_DIM)).to(device)
    dl = DataLoader(ds, batch_size=int(VAE_BATCH_SIZE), shuffle=True, drop_last=True)
    print(
        f"Training VAE on activations: N={acts.shape[0]:,} d={acts.shape[1]} "
        f"latent_dim={int(VAE_LATENT_DIM):,} hidden_dim={int(VAE_HIDDEN_DIM):,} beta={float(VAE_BETA)}"
    )
    logs = train_vae(
        vae=vae,
        loader=dl,
        device=device,
        lr=float(VAE_LR),
        beta=float(VAE_BETA),
        epochs=int(VAE_EPOCHS),
    )

    extractor_path = out_dir / f"vae_{ckpt_path.stem}_layer-{layer.replace('.', '_')}.pt"
    payload = {
        "extractor_type": "vae",
        "extractor_state_dict": vae.state_dict(),
        "extractor_class": type(vae).__name__,
        "extractor_config": {
            "d_in": int(vae.d_in),
            "feature_dim": int(vae.d_latent),
            "hidden_dim": int(vae.hidden_dim),
            "beta": float(VAE_BETA),
        },
        "source_checkpoint": str(ckpt_path),
        "source_model_config": model_cfg,
        "source_training_config": train_cfg_dict,
        "layer_name": layer,
        "include_cls_token": bool(INCLUDE_CLS_TOKEN),
        "max_activations": MAX_ACTIVATIONS,
        "disable_augmentations": bool(DISABLE_AUGMENTATIONS),
        "logs": logs,
    }
    torch.save(payload, extractor_path)
    print(f"Saved VAE checkpoint to: {extractor_path}")

else:
    # --- Train SVAE (SAE + VAE, summed reconstructions) ---
    try:
        from overcomplete.sae import TopKSAE
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Failed to import overcomplete. Install it with: pip install overcomplete") from e

    sae = TopKSAE(d_in, nb_concepts=int(SAE_NB_CONCEPTS), top_k=int(SAE_TOP_K), device=str(device))
    vae = ActivationVAE(d_in=d_in, d_latent=int(VAE_LATENT_DIM), hidden_dim=int(VAE_HIDDEN_DIM)).to(device)
    svae = ActivationSVAE(sae=sae, vae=vae).to(device)

    dl = DataLoader(ds, batch_size=int(SVAE_BATCH_SIZE), shuffle=True, drop_last=True)
    print(
        f"Training SVAE on activations: N={acts.shape[0]:,} d={acts.shape[1]} "
        f"sae_dim={int(SAE_NB_CONCEPTS):,} top_k={int(SAE_TOP_K)} | "
        f"vae_latent={int(VAE_LATENT_DIM):,} hidden_dim={int(VAE_HIDDEN_DIM):,} beta={float(SVAE_BETA)}"
    )
    logs = train_svae(
        svae=svae,
        loader=dl,
        device=device,
        lr=float(SVAE_LR),  # shared optimizer for both branches
        beta=float(SVAE_BETA),
        epochs=int(SVAE_EPOCHS),
    )

    extractor_path = out_dir / f"svae_{ckpt_path.stem}_layer-{layer.replace('.', '_')}.pt"
    payload = {
        "extractor_type": "svae",
        "extractor_state_dict": svae.state_dict(),
        "extractor_class": type(svae).__name__,
        "extractor_config": {
            "d_in": int(d_in),
            "feature_dim": int(SAE_NB_CONCEPTS + VAE_LATENT_DIM),
            "sae": {"nb_concepts": int(SAE_NB_CONCEPTS), "top_k": int(SAE_TOP_K)},
            "vae": {"latent_dim": int(VAE_LATENT_DIM), "hidden_dim": int(VAE_HIDDEN_DIM), "beta": float(SVAE_BETA)},
        },
        "source_checkpoint": str(ckpt_path),
        "source_model_config": model_cfg,
        "source_training_config": train_cfg_dict,
        "layer_name": layer,
        "include_cls_token": bool(INCLUDE_CLS_TOKEN),
        "max_activations": MAX_ACTIVATIONS,
        "disable_augmentations": bool(DISABLE_AUGMENTATIONS),
        "logs": logs,
    }
    torch.save(payload, extractor_path)
    print(f"Saved SVAE checkpoint to: {extractor_path}")


# %%
# Train an SAE after every ViT transformer block, then plot metrics.
if TRAIN_SAE_PER_VIT_BLOCK and str(model_cfg.get("model_type", "")).lower() == "vit":
    depth = int(model_cfg.get("depth", 0))
    if depth <= 0:
        raise ValueError(f"Invalid ViT depth in model_cfg: {model_cfg.get('depth')!r}")

    vit_layers: list[str] = [f"encoder.layers.{i}" for i in range(depth)]
    if VIT_BLOCK_LAYER_SET in ("blocks+final", "blocks+final+norm"):
        vit_layers.append("encoder")
    if VIT_BLOCK_LAYER_SET == "blocks+final+norm":
        vit_layers.append("norm")

    logs_by_layer: dict[str, dict[str, Any]] = {}

    for layer_name in vit_layers:
        print("\n" + "=" * 80)
        print(f"Training {extractor_type.upper()} for layer: {layer_name}")
        print("=" * 80)

        acts_i = collect_activations(
            model=model,
            loader=train_loader,
            layer_name=layer_name,
            device=device,
            max_activations=MAX_ACTIVATIONS,
            include_cls_token=bool(INCLUDE_CLS_TOKEN),
            progress_every=int(PROGRESS_EVERY),
        )

        d_in_i = int(acts_i.shape[1])
        ds_i = TensorDataset(acts_i)
        out_dir_i = out_dir / "per_layer"
        out_dir_i.mkdir(parents=True, exist_ok=True)

        if extractor_type == "sae":
            try:
                from overcomplete.sae import TopKSAE, train_sae
            except Exception as e:  # pragma: no cover
                raise RuntimeError("Failed to import overcomplete. Install it with: pip install overcomplete") from e

            sae_i = TopKSAE(d_in_i, nb_concepts=int(SAE_NB_CONCEPTS), top_k=int(SAE_TOP_K), device=str(device))
            dl_i = DataLoader(ds_i, batch_size=int(SAE_BATCH_SIZE), shuffle=True, drop_last=True)
            opt_i = torch.optim.Adam(sae_i.parameters(), lr=float(SAE_LR))
            print(
                f"Training SAE on activations: N={acts_i.shape[0]:,} d={acts_i.shape[1]} "
                f"nb_concepts={int(SAE_NB_CONCEPTS):,} top_k={int(SAE_TOP_K)}"
            )
            logs_i = train_sae(sae_i, dl_i, criterion, opt_i, nb_epochs=int(SAE_EPOCHS), device=str(device))
            logs_by_layer[layer_name] = dict(logs_i)

            extractor_path_i = out_dir_i / f"sae_{ckpt_path.stem}_layer-{layer_name.replace('.', '_')}.pt"
            payload_i = {
                # Back-compat
                "sae_state_dict": sae_i.state_dict(),
                "sae_class": type(sae_i).__name__,
                "sae_config": {
                    "d_in": d_in_i,
                    "nb_concepts": int(SAE_NB_CONCEPTS),
                    "top_k": int(SAE_TOP_K),
                },
                # Unified
                "extractor_type": "sae",
                "extractor_state_dict": sae_i.state_dict(),
                "extractor_class": type(sae_i).__name__,
                "extractor_config": {
                    "d_in": d_in_i,
                    "feature_dim": int(SAE_NB_CONCEPTS),
                    "top_k": int(SAE_TOP_K),
                },
                "source_checkpoint": str(ckpt_path),
                "source_model_config": model_cfg,
                "source_training_config": train_cfg_dict,
                "layer_name": layer_name,
                "include_cls_token": bool(INCLUDE_CLS_TOKEN),
                "max_activations": MAX_ACTIVATIONS,
                "disable_augmentations": bool(DISABLE_AUGMENTATIONS),
                "logs": logs_i,
            }
            torch.save(payload_i, extractor_path_i)
            print(f"Saved per-layer SAE checkpoint to: {extractor_path_i}")

        elif extractor_type == "vae":
            vae_i = ActivationVAE(d_in=d_in_i, d_latent=int(VAE_LATENT_DIM), hidden_dim=int(VAE_HIDDEN_DIM)).to(device)
            dl_i = DataLoader(ds_i, batch_size=int(VAE_BATCH_SIZE), shuffle=True, drop_last=True)
            print(
                f"Training VAE on activations: N={acts_i.shape[0]:,} d={acts_i.shape[1]} "
                f"latent_dim={int(VAE_LATENT_DIM):,} hidden_dim={int(VAE_HIDDEN_DIM):,} beta={float(VAE_BETA)}"
            )
            logs_i = train_vae(
                vae=vae_i,
                loader=dl_i,
                device=device,
                lr=float(VAE_LR),
                beta=float(VAE_BETA),
                epochs=int(VAE_EPOCHS),
            )
            logs_by_layer[layer_name] = dict(logs_i)

            extractor_path_i = out_dir_i / f"vae_{ckpt_path.stem}_layer-{layer_name.replace('.', '_')}.pt"
            payload_i = {
                "extractor_type": "vae",
                "extractor_state_dict": vae_i.state_dict(),
                "extractor_class": type(vae_i).__name__,
                "extractor_config": {
                    "d_in": int(vae_i.d_in),
                    "feature_dim": int(vae_i.d_latent),
                    "hidden_dim": int(vae_i.hidden_dim),
                    "beta": float(VAE_BETA),
                },
                "source_checkpoint": str(ckpt_path),
                "source_model_config": model_cfg,
                "source_training_config": train_cfg_dict,
                "layer_name": layer_name,
                "include_cls_token": bool(INCLUDE_CLS_TOKEN),
                "max_activations": MAX_ACTIVATIONS,
                "disable_augmentations": bool(DISABLE_AUGMENTATIONS),
                "logs": logs_i,
            }
            torch.save(payload_i, extractor_path_i)
            print(f"Saved per-layer VAE checkpoint to: {extractor_path_i}")

        else:
            # SVAE (SAE + VAE, summed reconstructions)
            try:
                from overcomplete.sae import TopKSAE
            except Exception as e:  # pragma: no cover
                raise RuntimeError("Failed to import overcomplete. Install it with: pip install overcomplete") from e

            sae_i = TopKSAE(d_in_i, nb_concepts=int(SAE_NB_CONCEPTS), top_k=int(SAE_TOP_K), device=str(device))
            vae_i = ActivationVAE(d_in=d_in_i, d_latent=int(VAE_LATENT_DIM), hidden_dim=int(VAE_HIDDEN_DIM)).to(device)
            svae_i = ActivationSVAE(sae=sae_i, vae=vae_i).to(device)
            dl_i = DataLoader(ds_i, batch_size=int(SVAE_BATCH_SIZE), shuffle=True, drop_last=True)

            print(
                f"Training SVAE on activations: N={acts_i.shape[0]:,} d={acts_i.shape[1]} "
                f"sae_dim={int(SAE_NB_CONCEPTS):,} top_k={int(SAE_TOP_K)} | "
                f"vae_latent={int(VAE_LATENT_DIM):,} hidden_dim={int(VAE_HIDDEN_DIM):,} beta={float(SVAE_BETA)}"
            )
            logs_i = train_svae(
                svae=svae_i,
                loader=dl_i,
                device=device,
                lr=float(SVAE_LR),
                beta=float(SVAE_BETA),
                epochs=int(SVAE_EPOCHS),
            )
            logs_by_layer[layer_name] = dict(logs_i)

            extractor_path_i = out_dir_i / f"svae_{ckpt_path.stem}_layer-{layer_name.replace('.', '_')}.pt"
            payload_i = {
                "extractor_type": "svae",
                "extractor_state_dict": svae_i.state_dict(),
                "extractor_class": type(svae_i).__name__,
                "extractor_config": {
                    "d_in": int(d_in_i),
                    "feature_dim": int(SAE_NB_CONCEPTS + VAE_LATENT_DIM),
                    "sae": {"nb_concepts": int(SAE_NB_CONCEPTS), "top_k": int(SAE_TOP_K)},
                    "vae": {
                        "latent_dim": int(VAE_LATENT_DIM),
                        "hidden_dim": int(VAE_HIDDEN_DIM),
                        "beta": float(SVAE_BETA),
                    },
                },
                "source_checkpoint": str(ckpt_path),
                "source_model_config": model_cfg,
                "source_training_config": train_cfg_dict,
                "layer_name": layer_name,
                "include_cls_token": bool(INCLUDE_CLS_TOKEN),
                "max_activations": MAX_ACTIVATIONS,
                "disable_augmentations": bool(DISABLE_AUGMENTATIONS),
                "logs": logs_i,
            }
            torch.save(payload_i, extractor_path_i)
            print(f"Saved per-layer SVAE checkpoint to: {extractor_path_i}")


    # --- Plot metrics ---
    def _plot_metric(key: str, *, title: str, ylabel: str, transform=None):
        plt.figure(figsize=(10, 5))
        for layer_name, l in logs_by_layer.items():
            y = l.get(key, [])
            if transform is not None:
                y = [transform(v) for v in y]
            plt.plot(y, label=layer_name)
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    if extractor_type == "sae":
        _plot_metric("avg_loss", title="SAE reconstruction loss by layer", ylabel="avg_loss")
        _plot_metric("r2", title="SAE reconstruction R2 by layer", ylabel="R2")
        _plot_metric(
            "dead_features",
            title="SAE dead feature % by layer",
            ylabel="dead_features (%)",
            transform=lambda v: float(v) * 100.0,
        )
    elif extractor_type == "vae":
        _plot_metric("loss/total", title="VAE total loss by layer", ylabel="loss/total")
        _plot_metric("loss/recon", title="VAE reconstruction loss by layer", ylabel="loss/recon")
        _plot_metric("loss/kl", title="VAE KL divergence loss by layer", ylabel="loss/kl")
        _plot_metric("r2", title="VAE reconstruction R2 by layer", ylabel="R2")
    else:
        _plot_metric("loss/total", title="SVAE total loss by layer", ylabel="loss/total")
        _plot_metric("loss/recon", title="SVAE reconstruction loss by layer", ylabel="loss/recon")
        _plot_metric("loss/kl", title="SVAE KL divergence loss by layer", ylabel="loss/kl")
        _plot_metric("r2", title="SVAE reconstruction R2 by layer", ylabel="R2")
        _plot_metric(
            "dead_features",
            title="SVAE dead feature % by layer",
            ylabel="dead_features (%)",
            transform=lambda v: float(v) * 100.0,
        )



