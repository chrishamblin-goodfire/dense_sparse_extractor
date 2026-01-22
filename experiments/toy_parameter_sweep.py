"""
Executable toy experiment runner.

What this script does:
- Generates a synthetic dense+sparse toy dataset X = X_sparse + X_dense (+ optional noise)
- Sweeps over:
  - CONT_MODE in {"subspace_uniform", "subspace_gaussian", "sphere"}
  - encoder models in {MpSAE, SAE (vanilla), TopKSAE, ToyDenSaE, DenseVAEPlusSparse}
  - small per-model hyperparameter grids
- Saves artifacts to:
  toy_experiments_results/<data_source>/<model>/<model_hparams>/<seed_x>/...
  including:
    - toy data tensors
    - trained model weights + init kwargs
    - snapshot of this run file
    - config + basic metrics

Run:
  python dense_sparse_extractor/experiments/toy_parameter_sweep.py
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha1
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from overcomplete.sae import MpSAE, SAE, TopKSAE, mse_l1, train_sae

    HAVE_OVERCOMPLETE = True
except Exception:
    HAVE_OVERCOMPLETE = False


# =========================
# Experiment config (edit)
# =========================

# Results folder will be created at repo root (next to `experiments/`, `notebooks/`, etc).
RESULTS_ROOT = (Path(__file__).resolve().parents[1] / "toy_experiments_results").resolve()

# Repro
BASE_SEED = 0
SEEDS = [0]  # add more for repeats, e.g. [0,1,2]

# Device: training defaults to CUDA if available; if you want CPU-only, set "cpu".
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Parallelism
ENABLE_PARALLEL = True
MAX_WORKERS = 6
# If you're on a big GPU (like H200) and these jobs are small, GPU multiprocessing can help throughput.
# Warning: all workers will contend for the same GPU unless you manually shard GPUs via CUDA_VISIBLE_DEVICES.
MAX_WORKERS_CUDA = 4

# Dataset sweep
CONT_MODES: list[Literal["subspace_uniform", "subspace_gaussian", "sphere"]] = [
    "subspace_uniform",
    "subspace_gaussian",
    "sphere",
]

# Model sweep toggles
RUN_MODELS = {
    "vanilla_sae": True,
    "topk_sae": True,
    "mp_sae": True,
    "densae": True,
    "dense_vae_plus_sparse": True,
}

# Save extras
SAVE_FULL_DATASET = True


# =========================
# Data + training configs
# =========================


@dataclass(frozen=True)
class DataConfig:
    # Ambient embedding dim
    d: int = 10

    # Dataset size and batching for generation
    n: int = 50_000
    gen_batch: int = 4096

    # Sparse component
    n_sparse: int = 50
    sparse_mode: Literal["bernoulli", "topk"] = "bernoulli"
    sparse_bernoulli_p: float = 0.04
    sparse_topk_k: int = 3
    sparse_scale: float = 1.0

    # Dense component
    cont_mode: Literal["subspace_uniform", "subspace_gaussian", "sphere"] = "sphere"
    cont_dim: int = 3
    cont_scale: float = 1.0

    # Noise
    obs_noise_std: float = 0.0

    # Direction construction
    orthogonalize_sparse_dirs: bool = True
    orthogonalize_cont_basis: bool = True
    norm_jitter_std: float = 0.05

    def data_source_id(self) -> str:
        # Keep readable but deterministic.
        parts = [
            f"D{self.d}",
            f"N{self.n}",
            f"sparse-{self.sparse_mode}",
            f"ns{self.n_sparse}",
            f"p{self.sparse_bernoulli_p:g}" if self.sparse_mode == "bernoulli" else f"k{self.sparse_topk_k}",
            f"ss{self.sparse_scale:g}",
            f"cont-{self.cont_mode}",
            f"cd{self.cont_dim}",
            f"cs{self.cont_scale:g}",
            f"noise{self.obs_noise_std:g}",
        ]
        return "__".join(parts)


@dataclass(frozen=True)
class SharedTrainConfig:
    # train subset size (speed knob)
    train_n: int = 20_480
    batch_size: int = 1024
    epochs: int = 100
    lr: float = 3e-3
    standardize_x: bool = True


@dataclass(frozen=True)
class SAEConfig:
    n_concepts: Optional[int] = None  # default: 2 * n_sparse
    l1_penalty: float = 2e-4
    epochs: int = 300
    lr: float = 5e-4
    batch_size: int = 500
    standardize_x: bool = True


@dataclass(frozen=True)
class TopKSAEConfig:
    n_concepts: Optional[int] = None  # default: 2 * n_sparse
    top_k: Optional[int] = None  # default: derived from expected sparsity
    epochs: int = 300
    lr: float = 5e-4
    batch_size: int = 500
    standardize_x: bool = True


@dataclass(frozen=True)
class MpSAEConfig:
    n_concepts: Optional[int] = None  # default: 2 * n_sparse
    k: int = 3
    epochs: int = 300
    lr: float = 5e-4
    batch_size: int = 500
    standardize_x: bool = True


@dataclass(frozen=True)
class DenSaEConfig:
    d_dense: Optional[int] = None  # default: cont_dim
    d_sparse: Optional[int] = None  # default: 2*n_sparse
    n_iters: int = 15
    alpha_z: float = 0.25
    alpha_s: float = 0.25
    thresh: float = 0.10
    twosided: bool = False
    inv_lambda_x_update: float = 0.0
    dense_reg_weight_loss: float = 0.0
    train: SharedTrainConfig = SharedTrainConfig()


@dataclass(frozen=True)
class DenseVAEPlusSparseConfig:
    z_dim: Optional[int] = None  # default: cont_dim
    d_sparse: Optional[int] = None  # default: 2*n_sparse
    hidden: int = 128
    sparse_thresh: float = 0.10
    twosided_sparse: bool = False
    lambda_s: float = 1e-3
    kl_beta_max: float = 1e-3
    kl_warmup_steps: int = 0
    kl_ramp_steps: int = 10_000
    kl_schedule: Literal["constant", "linear"] = "linear"
    train: SharedTrainConfig = SharedTrainConfig()


# =========================
# Core utilities
# =========================


def _now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def qr_orthonormal_rows(n_rows: int, n_cols: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    if n_rows > n_cols:
        raise ValueError(f"Need n_rows <= n_cols for orthonormal rows, got {n_rows} > {n_cols}.")
    q, _ = torch.linalg.qr(torch.randn(n_cols, n_cols, device=device, dtype=dtype))
    return q[:n_rows]


def maybe_jitter_norms(v: torch.Tensor, std: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if std <= 0:
        scales = torch.ones(v.shape[0], device=v.device, dtype=v.dtype)
        return v, scales
    scales = 1.0 + torch.randn(v.shape[0], device=v.device, dtype=v.dtype) * std
    return v * scales[:, None], scales


def sample_sparse_acts(
    *,
    batch: int,
    n_sparse: int,
    mode: Literal["bernoulli", "topk"],
    p: float,
    k: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if mode == "bernoulli":
        return (torch.rand(batch, n_sparse, device=device) < p).to(dtype)
    if mode == "topk":
        if k <= 0 or k > n_sparse:
            raise ValueError(f"Need 1 <= k <= n_sparse; got k={k}, n_sparse={n_sparse}")
        idx = torch.randint(low=0, high=n_sparse, size=(batch, k), device=device)
        acts = torch.zeros(batch, n_sparse, device=device, dtype=dtype)
        acts.scatter_(dim=1, index=idx, value=1.0)
        return acts
    raise ValueError(f"Unknown sparse mode: {mode}")


def _sphere(batch: int, dim: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    if dim < 2:
        raise ValueError("sphere requires cont_dim >= 2")
    z = torch.randn(batch, dim, device=device, dtype=dtype)
    return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)


def sample_continuous_latents(
    *,
    batch: int,
    mode: Literal["subspace_uniform", "subspace_gaussian", "sphere"],
    cont_dim: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if mode == "subspace_uniform":
        return (torch.rand(batch, cont_dim, device=device, dtype=dtype) * 2.0) - 1.0
    if mode == "subspace_gaussian":
        return torch.randn(batch, cont_dim, device=device, dtype=dtype)
    if mode == "sphere":
        return _sphere(batch, cont_dim, device=device, dtype=dtype)
    raise ValueError(f"Unknown continuous mode: {mode}")


def embed_continuous(z: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if z.shape[1] != basis.shape[0]:
        raise ValueError(f"z has dim {z.shape[1]} but basis has {basis.shape[0]} rows")
    return z @ basis


@torch.no_grad()
def generate_toy_dataset(cfg: DataConfig, *, seed: int, device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    set_seed(seed)

    # Sparse directions: [n_sparse, d]
    if cfg.orthogonalize_sparse_dirs and cfg.n_sparse <= cfg.d:
        sparse_dirs = qr_orthonormal_rows(cfg.n_sparse, cfg.d, device=device, dtype=dtype)
    else:
        sparse_dirs = torch.randn(cfg.n_sparse, cfg.d, device=device, dtype=dtype)
        sparse_dirs = sparse_dirs / sparse_dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sparse_dirs, sparse_dir_scales = maybe_jitter_norms(sparse_dirs, cfg.norm_jitter_std)

    # Continuous basis: [cont_dim, d]
    if cfg.orthogonalize_cont_basis and cfg.cont_dim <= cfg.d:
        cont_basis = qr_orthonormal_rows(cfg.cont_dim, cfg.d, device=device, dtype=dtype)
    else:
        cont_basis = torch.randn(cfg.cont_dim, cfg.d, device=device, dtype=dtype)
        cont_basis = cont_basis / cont_basis.norm(dim=1, keepdim=True).clamp_min(1e-12)

    xs: list[torch.Tensor] = []
    ss: list[torch.Tensor] = []
    zs: list[torch.Tensor] = []

    for start in range(0, cfg.n, cfg.gen_batch):
        b = min(cfg.gen_batch, cfg.n - start)
        s = sample_sparse_acts(
            batch=b,
            n_sparse=cfg.n_sparse,
            mode=cfg.sparse_mode,
            p=cfg.sparse_bernoulli_p,
            k=cfg.sparse_topk_k,
            device=device,
            dtype=dtype,
        )
        z = sample_continuous_latents(
            batch=b,
            mode=cfg.cont_mode,
            cont_dim=cfg.cont_dim,
            device=device,
            dtype=dtype,
        )
        x_sparse = (s @ sparse_dirs) * cfg.sparse_scale
        x_cont = embed_continuous(z, cont_basis) * cfg.cont_scale
        x = x_sparse + x_cont
        if cfg.obs_noise_std > 0:
            x = x + torch.randn_like(x) * cfg.obs_noise_std

        xs.append(x.detach().cpu())
        ss.append(s.detach().cpu())
        zs.append(z.detach().cpu())

    X = torch.cat(xs, dim=0)
    S = torch.cat(ss, dim=0)
    Z = torch.cat(zs, dim=0)

    return {
        "X": X,
        "S": S,
        "Z": Z,
        "sparse_dirs": sparse_dirs.detach().cpu(),
        "sparse_dir_scales": sparse_dir_scales.detach().cpu(),
        "cont_basis": cont_basis.detach().cpu(),
    }


def _standardize_train_full(X: torch.Tensor, train_n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X_train = X[:train_n]
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return X_train, mean, std, (X - mean) / std


def _make_run_dir(
    *,
    data_source: str,
    model_name: str,
    model_hparams: Dict[str, Any],
    seed: int,
) -> Path:
    hp_json = json.dumps(model_hparams, sort_keys=True, separators=(",", ":"))
    hp_hash = sha1(hp_json.encode("utf-8")).hexdigest()[:12]
    hp_readable = "__".join([f"{k}={model_hparams[k]}" for k in sorted(model_hparams.keys())])
    hp_dirname = f"{hp_readable}__h{hp_hash}" if hp_readable else f"h{hp_hash}"

    run_dir = RESULTS_ROOT / data_source / model_name / hp_dirname / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_snapshot(run_dir: Path) -> None:
    snapshot_path = run_dir / "run_snapshot.py"
    snapshot_path.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")


def _runtime_info() -> Dict[str, Any]:
    return {
        "utc_time": _now_utc_compact(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": DEVICE,
        "cwd": str(Path.cwd()),
        "script": str(Path(__file__).resolve()),
    }


# =========================
# Models
# =========================


def soft_shrink(x: torch.Tensor, thresh: float, twosided: bool) -> torch.Tensor:
    if twosided:
        return torch.sign(x) * torch.relu(torch.abs(x) - thresh)
    return torch.relu(x - thresh)


class ToyDenSaE(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_dense: int,
        d_sparse: int,
        *,
        n_iters: int = 10,
        alpha_z: float = 0.2,
        alpha_s: float = 0.2,
        thresh: float = 0.1,
        twosided: bool = False,
        inv_lambda_x: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.d_dense = int(d_dense)
        self.d_sparse = int(d_sparse)
        self.n_iters = int(n_iters)
        self.alpha_z = float(alpha_z)
        self.alpha_s = float(alpha_s)
        self.thresh = float(thresh)
        self.twosided = bool(twosided)
        self.inv_lambda_x = float(inv_lambda_x)
        self.device_str = str(device)

        A0 = torch.randn(self.d_dense, self.d_in, device=device)
        B0 = torch.randn(self.d_sparse, self.d_in, device=device)
        A0 = A0 / A0.norm(dim=1, keepdim=True).clamp_min(1e-12)
        B0 = B0 / B0.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.A = nn.Parameter(A0)
        self.B = nn.Parameter(B0)

    @torch.no_grad()
    def normalize_dicts_(self) -> None:
        self.A.data = self.A.data / self.A.data.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.B.data = self.B.data / self.B.data.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def decode(self, z: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_dense = z @ self.A
        x_sparse = s @ self.B
        return x_dense + x_sparse, x_dense, x_sparse

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(x.shape[0], self.d_dense, device=x.device, dtype=x.dtype)
        s = torch.zeros(x.shape[0], self.d_sparse, device=x.device, dtype=x.dtype)
        for _ in range(self.n_iters):
            x_hat, x_dense, _ = self.decode(z, s)
            res = x - x_hat
            z = z + self.alpha_z * (res @ self.A.T)
            if self.inv_lambda_x > 0:
                z = z - self.alpha_z * (self.inv_lambda_x * (x_dense @ self.A.T))
            s_pre = s + self.alpha_s * (res @ self.B.T)
            s = soft_shrink(s_pre, self.thresh, self.twosided)
        return z, s

    def forward(self, x: torch.Tensor):
        z, s = self.encode(x)
        x_hat, x_dense, x_sparse = self.decode(z, s)
        return x_hat, z, s, x_dense, x_sparse


class DenseVAEPlusSparse(nn.Module):
    def __init__(
        self,
        d_in: int,
        z_dim: int,
        d_sparse: int,
        *,
        hidden: int = 128,
        twosided_sparse: bool = False,
        sparse_thresh: float = 0.1,
        kl_beta_max: float = 1e-3,
        kl_warmup_steps: int = 0,
        kl_ramp_steps: int = 10_000,
        kl_schedule: Literal["constant", "linear"] = "linear",
        device: str = "cpu",
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.z_dim = int(z_dim)
        self.d_sparse = int(d_sparse)
        self.hidden = int(hidden)
        self.twosided_sparse = bool(twosided_sparse)
        self.sparse_thresh = float(sparse_thresh)

        self.kl_beta_max = float(kl_beta_max)
        self.kl_warmup_steps = int(kl_warmup_steps)
        self.kl_ramp_steps = int(kl_ramp_steps)
        self.kl_schedule = kl_schedule

        self.enc = nn.Sequential(
            nn.Linear(self.d_in, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(self.hidden, self.z_dim)
        self.enc_logvar = nn.Linear(self.hidden, self.z_dim)

        self.dec = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.d_in),
        )

        self.sparse_enc = nn.Linear(self.d_in, self.d_sparse, bias=True)
        B0 = torch.randn(self.d_sparse, self.d_in, device=device)
        B0 = B0 / B0.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.B = nn.Parameter(B0)

    def kl_beta(self, step: int) -> float:
        if self.kl_schedule == "constant":
            return self.kl_beta_max
        if self.kl_schedule != "linear":
            raise ValueError(f"Unknown kl_schedule: {self.kl_schedule}")
        if step < self.kl_warmup_steps:
            return 0.0
        if self.kl_ramp_steps <= 0:
            return self.kl_beta_max
        t = (step - self.kl_warmup_steps) / float(self.kl_ramp_steps)
        t = max(0.0, min(1.0, t))
        return self.kl_beta_max * t

    @torch.no_grad()
    def normalize_dict_(self) -> None:
        self.B.data = self.B.data / self.B.data.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def encode_dense(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h).clamp(min=-20.0, max=10.0)
        return mu, logvar

    def sample_z(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode_dense(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def encode_sparse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pre = self.sparse_enc(x)
        s = soft_shrink(pre, self.sparse_thresh, self.twosided_sparse)
        return pre, s

    def decode_sparse(self, s: torch.Tensor) -> torch.Tensor:
        return s @ self.B

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode_dense(x)
        z = self.sample_z(mu, logvar)
        x_dense = self.decode_dense(z)
        pre_s, s = self.encode_sparse(x)
        x_sparse = self.decode_sparse(s)
        x_hat = x_dense + x_sparse
        return x_hat, (mu, logvar, z), (pre_s, s), (x_dense, x_sparse)


def kl_diag_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1).mean()


# =========================
# Training wrappers
# =========================


def _train_overcomplete_sae_family(
    *,
    model: nn.Module,
    X: torch.Tensor,
    train_n: int,
    batch_size: int,
    epochs: int,
    lr: float,
    standardize_x: bool,
    device: str,
    l1_penalty: Optional[float] = None,
) -> Dict[str, Any]:
    if not HAVE_OVERCOMPLETE:
        raise RuntimeError("Overcomplete is not available; cannot train SAE/TopKSAE/MpSAE.")

    X_train = X[:train_n]
    if standardize_x:
        X_train, mean, std, X_full_in = _standardize_train_full(X, train_n)
        X_train_in = X_train
    else:
        mean = torch.zeros(1, X.shape[1])
        std = torch.ones(1, X.shape[1])
        X_train_in = X_train
        X_full_in = X

    train_loader = DataLoader(
        TensorDataset(X_train_in),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if l1_penalty is None:
        criterion = lambda x, x_hat, pre, z, d: (x - x_hat).square().mean()
    else:
        criterion = lambda x, x_hat, pre, z, d: mse_l1(x, x_hat, pre, z, d, penalty=l1_penalty)

    t0 = time.time()
    _ = train_sae(model, train_loader, criterion, opt, nb_epochs=epochs, device=device)
    train_seconds = time.time() - t0

    # Basic metrics: recon on a small batch
    model.eval()
    with torch.no_grad():
        xb = X_full_in[: min(2048, X_full_in.shape[0])].to(device)
        recon = float("nan")
        try:
            x_hat, _z = model.encode(xb)
            recon = float((xb - x_hat).square().mean().item())
        except Exception:
            pass

        # Attempt to compute sparsity via encode()
        l0 = float("nan")
        try:
            _x_hat2, z2 = model.encode(xb)
            l0 = float((z2.detach() > 0).float().sum(dim=1).mean().item())
        except Exception:
            pass

    return {
        "train_seconds": train_seconds,
        "recon_mse_smallbatch": recon,
        "avg_l0_smallbatch": l0,
        "x_mean": mean,
        "x_std": std,
    }


def train_vanilla_sae(
    *,
    X: torch.Tensor,
    n_sparse: int,
    d_in: int,
    cfg: SAEConfig,
    device: str,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    n_concepts = int(cfg.n_concepts or (2 * n_sparse))
    model = SAE(d_in, n_concepts, device=device)
    metrics = _train_overcomplete_sae_family(
        model=model,
        X=X,
        train_n=min(20_000, X.shape[0]),
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        standardize_x=cfg.standardize_x,
        device=device,
        l1_penalty=cfg.l1_penalty,
    )
    init_kwargs = {"d_in": d_in, "n_concepts": n_concepts, "device": device}
    hparams = asdict(cfg)
    return model, metrics, {"init_kwargs": init_kwargs, "hparams": hparams}


def train_topk_sae(
    *,
    X: torch.Tensor,
    n_sparse: int,
    d_in: int,
    cfg: TopKSAEConfig,
    expected_l0: float,
    device: str,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    n_concepts = int(cfg.n_concepts or (2 * n_sparse))
    top_k = int(cfg.top_k or max(1, 2 * math.ceil(expected_l0)))
    model = TopKSAE(d_in, n_concepts, top_k=top_k, device=device)
    metrics = _train_overcomplete_sae_family(
        model=model,
        X=X,
        train_n=min(20_000, X.shape[0]),
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        standardize_x=cfg.standardize_x,
        device=device,
        l1_penalty=None,
    )
    init_kwargs = {"d_in": d_in, "n_concepts": n_concepts, "top_k": top_k, "device": device}
    hparams = asdict(cfg)
    return model, metrics, {"init_kwargs": init_kwargs, "hparams": hparams}


def train_mp_sae(
    *,
    X: torch.Tensor,
    n_sparse: int,
    d_in: int,
    cfg: MpSAEConfig,
    device: str,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    n_concepts = int(cfg.n_concepts or (2 * n_sparse))
    model = MpSAE(d_in, n_concepts, k=int(cfg.k), device=device)
    metrics = _train_overcomplete_sae_family(
        model=model,
        X=X,
        train_n=min(20_000, X.shape[0]),
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        standardize_x=cfg.standardize_x,
        device=device,
        l1_penalty=None,
    )
    init_kwargs = {"d_in": d_in, "n_concepts": n_concepts, "k": int(cfg.k), "device": device}
    hparams = asdict(cfg)
    return model, metrics, {"init_kwargs": init_kwargs, "hparams": hparams}


def train_densae(
    *,
    X: torch.Tensor,
    d_in: int,
    cont_dim: int,
    n_sparse: int,
    cfg: DenSaEConfig,
    device: str,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    d_dense = int(cfg.d_dense or max(2, cont_dim))
    d_sparse = int(cfg.d_sparse or (2 * n_sparse))

    train_cfg = cfg.train
    train_n = min(train_cfg.train_n, X.shape[0])
    X_train = X[:train_n].to(device)

    if train_cfg.standardize_x:
        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        X_train_in = (X_train - mean) / std
    else:
        mean = torch.zeros(1, d_in, device=device)
        std = torch.ones(1, d_in, device=device)
        X_train_in = X_train

    loader = DataLoader(
        TensorDataset(X_train_in.detach().cpu()),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    model = ToyDenSaE(
        d_in=d_in,
        d_dense=d_dense,
        d_sparse=d_sparse,
        n_iters=cfg.n_iters,
        alpha_z=cfg.alpha_z,
        alpha_s=cfg.alpha_s,
        thresh=cfg.thresh,
        twosided=cfg.twosided,
        inv_lambda_x=cfg.inv_lambda_x_update,
        device=device,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    t0 = time.time()
    for _epoch in range(train_cfg.epochs):
        model.train()
        for (xb_cpu,) in loader:
            xb = xb_cpu.to(device, non_blocking=True)
            x_hat, _z, s, x_dense, _x_sparse = model(xb)
            recon = (xb - x_hat).square().mean()
            sparse_pen = s.abs().mean()
            dense_pen = x_dense.square().mean()
            loss = recon + (1e-3 * sparse_pen) + (cfg.dense_reg_weight_loss * dense_pen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                model.normalize_dicts_()
    train_seconds = time.time() - t0

    model.eval()
    with torch.no_grad():
        xb = X_train_in[: min(2048, X_train_in.shape[0])].to(device)
        x_hat, _z, s, *_ = model(xb)
        recon = float((xb - x_hat).square().mean().item())
        l0 = float((s.abs() > 1e-8).float().sum(dim=1).mean().item())

    init_kwargs = {
        "d_in": d_in,
        "d_dense": d_dense,
        "d_sparse": d_sparse,
        "n_iters": cfg.n_iters,
        "alpha_z": cfg.alpha_z,
        "alpha_s": cfg.alpha_s,
        "thresh": cfg.thresh,
        "twosided": cfg.twosided,
        "inv_lambda_x": cfg.inv_lambda_x_update,
        "device": device,
    }
    hparams = asdict(cfg)
    return (
        model,
        {
            "train_seconds": train_seconds,
            "recon_mse_smallbatch": recon,
            "avg_l0_smallbatch": l0,
            "x_mean": mean.detach().cpu(),
            "x_std": std.detach().cpu(),
        },
        {"init_kwargs": init_kwargs, "hparams": hparams},
    )


def train_dense_vae_plus_sparse(
    *,
    X: torch.Tensor,
    d_in: int,
    cont_dim: int,
    n_sparse: int,
    cfg: DenseVAEPlusSparseConfig,
    device: str,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    z_dim = int(cfg.z_dim or max(2, cont_dim))
    d_sparse = int(cfg.d_sparse or (2 * n_sparse))
    train_cfg = cfg.train
    train_n = min(train_cfg.train_n, X.shape[0])
    X_train = X[:train_n].to(device)

    if train_cfg.standardize_x:
        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        X_train_in = (X_train - mean) / std
    else:
        mean = torch.zeros(1, d_in, device=device)
        std = torch.ones(1, d_in, device=device)
        X_train_in = X_train

    loader = DataLoader(
        TensorDataset(X_train_in.detach().cpu()),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    model = DenseVAEPlusSparse(
        d_in=d_in,
        z_dim=z_dim,
        d_sparse=d_sparse,
        hidden=cfg.hidden,
        twosided_sparse=cfg.twosided_sparse,
        sparse_thresh=cfg.sparse_thresh,
        kl_beta_max=cfg.kl_beta_max,
        kl_warmup_steps=cfg.kl_warmup_steps,
        kl_ramp_steps=cfg.kl_ramp_steps,
        kl_schedule=cfg.kl_schedule,
        device=device,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    global_step = 0
    t0 = time.time()
    for _epoch in range(train_cfg.epochs):
        model.train()
        for (xb_cpu,) in loader:
            xb = xb_cpu.to(device, non_blocking=True)
            x_hat, (mu, logvar, _z), (_pre_s, s), (_xd, _xs) = model(xb)
            recon = (xb - x_hat).square().mean()
            kl = kl_diag_gaussian(mu, logvar)
            sparse_pen = s.abs().mean()
            beta = model.kl_beta(global_step)
            loss = recon + (beta * kl) + (cfg.lambda_s * sparse_pen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                model.normalize_dict_()
            global_step += 1
    train_seconds = time.time() - t0

    model.eval()
    with torch.no_grad():
        xb = X_train_in[: min(2048, X_train_in.shape[0])].to(device)
        x_hat, (mu, logvar, _z), (_pre_s, s), _ = model(xb)
        recon = float((xb - x_hat).square().mean().item())
        klv = float(kl_diag_gaussian(mu, logvar).item())
        l0 = float((s.abs() > 1e-8).float().sum(dim=1).mean().item())

    init_kwargs = {
        "d_in": d_in,
        "z_dim": z_dim,
        "d_sparse": d_sparse,
        "hidden": cfg.hidden,
        "twosided_sparse": cfg.twosided_sparse,
        "sparse_thresh": cfg.sparse_thresh,
        "kl_beta_max": cfg.kl_beta_max,
        "kl_warmup_steps": cfg.kl_warmup_steps,
        "kl_ramp_steps": cfg.kl_ramp_steps,
        "kl_schedule": cfg.kl_schedule,
        "device": device,
    }
    hparams = asdict(cfg)
    return (
        model,
        {
            "train_seconds": train_seconds,
            "recon_mse_smallbatch": recon,
            "kl_smallbatch": klv,
            "avg_l0_smallbatch": l0,
            "x_mean": mean.detach().cpu(),
            "x_std": std.detach().cpu(),
        },
        {"init_kwargs": init_kwargs, "hparams": hparams},
    )


# =========================
# Sweep definition
# =========================


def build_model_sweeps(data_cfg: DataConfig) -> list[Tuple[str, Dict[str, Any]]]:
    """
    Returns list of (model_name, model_hparams_dict) where hparams dict is used for foldering
    and to build the actual model config.
    """
    out: list[Tuple[str, Dict[str, Any]]] = []

    expected_l0 = (
        data_cfg.n_sparse * data_cfg.sparse_bernoulli_p
        if data_cfg.sparse_mode == "bernoulli"
        else float(data_cfg.sparse_topk_k)
    )
    base_k = max(1, 2 * math.ceil(expected_l0))

    # Vanilla SAE: sweep L1 a bit
    if RUN_MODELS.get("vanilla_sae", False):
        for l1 in [1e-4, 2e-4, 1e-3]:
            out.append(("vanilla_sae", {"l1_penalty": l1, "n_concepts": 2 * data_cfg.n_sparse}))

    # TopK: sweep top_k around expected sparsity
    if RUN_MODELS.get("topk_sae", False):
        for top_k in sorted(set([base_k, 2 * base_k, 10 * base_k])):
            out.append(("topk_sae", {"top_k": int(top_k), "n_concepts": 2 * data_cfg.n_sparse}))

    # MpSAE: sweep k steps
    if RUN_MODELS.get("mp_sae", False):
        for k in [base_k, 2 * base_k, 10 * base_k]:
            out.append(("mp_sae", {"k": int(k), "n_concepts": 2 * data_cfg.n_sparse}))

    # DenSaE: sweep threshold + n_iters lightly
    if RUN_MODELS.get("densae", False):
        for thresh, n_iters in product([0.05, 0.10], [10, 15]):
            out.append(
                (
                    "densae",
                    {
                        "d_dense": max(2, data_cfg.cont_dim),
                        "d_sparse": 2 * data_cfg.n_sparse,
                        "thresh": float(thresh),
                        "n_iters": int(n_iters),
                        "inv_lambda_x_update": 0.0,
                    },
                )
            )

    # DenseVAEPlusSparse: sweep KL beta and sparse threshold lightly
    if RUN_MODELS.get("dense_vae_plus_sparse", False):
        for kl_beta_max, sparse_thresh in product([1e-4, 1e-3], [0.05, 0.10]):
            out.append(
                (
                    "dense_vae_plus_sparse",
                    {
                        "z_dim": max(2, data_cfg.cont_dim),
                        "d_sparse": 2 * data_cfg.n_sparse,
                        "kl_beta_max": float(kl_beta_max),
                        "sparse_thresh": float(sparse_thresh),
                        "lambda_s": 1e-3,
                    },
                )
            )

    return out


def build_jobs() -> list[Dict[str, Any]]:
    jobs: list[Dict[str, Any]] = []

    for cont_mode in CONT_MODES:
        data_cfg = DataConfig(cont_mode=cont_mode)
        model_sweeps = build_model_sweeps(data_cfg)
        for seed in SEEDS:
            for model_name, model_hp in model_sweeps:
                jobs.append(
                    {
                        "data_cfg": asdict(data_cfg),
                        "model_name": model_name,
                        "model_hp": model_hp,
                        "seed": int(seed),
                    }
                )
    return jobs


# =========================
# Job execution
# =========================


def _run_one_job(job: Dict[str, Any]) -> str:
    """
    Worker-safe function. Returns a short status string.
    """
    # Reduce CPU contention in multi-process runs
    torch.set_num_threads(1)
    if DEVICE != "cpu" and torch.cuda.is_available():
        # Make device choice explicit inside subprocesses.
        torch.cuda.set_device(0)

    seed = int(job["seed"])
    data_cfg = DataConfig(**job["data_cfg"])
    model_name = str(job["model_name"])
    model_hp: Dict[str, Any] = dict(job["model_hp"])

    # Derive expected sparsity for TopK defaults (if needed)
    expected_l0 = (
        data_cfg.n_sparse * data_cfg.sparse_bernoulli_p
        if data_cfg.sparse_mode == "bernoulli"
        else float(data_cfg.sparse_topk_k)
    )

    run_dir = _make_run_dir(
        data_source=data_cfg.data_source_id(),
        model_name=model_name,
        model_hparams=model_hp,
        seed=seed,
    )

    # If re-running, keep old outputs by nesting a timestamped "attempt" dir.
    attempt_dir = run_dir / f"attempt_{_now_utc_compact()}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot + config
    _save_snapshot(attempt_dir)
    _write_json(
        attempt_dir / "config.json",
        {
            "runtime": _runtime_info(),
            "seed": seed,
            "data_cfg": asdict(data_cfg),
            "model_name": model_name,
            "model_hp": model_hp,
        },
    )

    # Generate data (always on CPU tensors returned)
    data = generate_toy_dataset(data_cfg, seed=BASE_SEED + seed, device=DEVICE, dtype=DTYPE)
    X = data["X"]  # CPU

    if SAVE_FULL_DATASET:
        # Store S as uint8 to reduce size (binary in current configs)
        S_u8 = (data["S"] > 0).to(torch.uint8)
        torch.save(
            {
                "X": data["X"],
                "S_u8": S_u8,
                "Z": data["Z"],
                "sparse_dirs": data["sparse_dirs"],
                "sparse_dir_scales": data["sparse_dir_scales"],
                "cont_basis": data["cont_basis"],
            },
            attempt_dir / "toy_data.pt",
        )

    model: nn.Module
    model_metrics: Dict[str, Any]
    model_meta: Dict[str, Any]

    if model_name in {"vanilla_sae", "topk_sae", "mp_sae"} and not HAVE_OVERCOMPLETE:
        raise RuntimeError(
            f"Requested {model_name}, but Overcomplete isn't importable in this environment."
        )

    if model_name == "vanilla_sae":
        cfg = SAEConfig(n_concepts=int(model_hp["n_concepts"]), l1_penalty=float(model_hp["l1_penalty"]))
        model, model_metrics, model_meta = train_vanilla_sae(
            X=X,
            n_sparse=data_cfg.n_sparse,
            d_in=data_cfg.d,
            cfg=cfg,
            device=DEVICE,
        )
    elif model_name == "topk_sae":
        cfg = TopKSAEConfig(n_concepts=int(model_hp["n_concepts"]), top_k=int(model_hp["top_k"]))
        model, model_metrics, model_meta = train_topk_sae(
            X=X,
            n_sparse=data_cfg.n_sparse,
            d_in=data_cfg.d,
            cfg=cfg,
            expected_l0=expected_l0,
            device=DEVICE,
        )
    elif model_name == "mp_sae":
        cfg = MpSAEConfig(n_concepts=int(model_hp["n_concepts"]), k=int(model_hp["k"]))
        model, model_metrics, model_meta = train_mp_sae(
            X=X,
            n_sparse=data_cfg.n_sparse,
            d_in=data_cfg.d,
            cfg=cfg,
            device=DEVICE,
        )
    elif model_name == "densae":
        cfg = DenSaEConfig(
            d_dense=int(model_hp["d_dense"]),
            d_sparse=int(model_hp["d_sparse"]),
            n_iters=int(model_hp["n_iters"]),
            thresh=float(model_hp["thresh"]),
            inv_lambda_x_update=float(model_hp.get("inv_lambda_x_update", 0.0)),
        )
        model, model_metrics, model_meta = train_densae(
            X=X,
            d_in=data_cfg.d,
            cont_dim=data_cfg.cont_dim,
            n_sparse=data_cfg.n_sparse,
            cfg=cfg,
            device=DEVICE,
        )
    elif model_name == "dense_vae_plus_sparse":
        cfg = DenseVAEPlusSparseConfig(
            z_dim=int(model_hp["z_dim"]),
            d_sparse=int(model_hp["d_sparse"]),
            kl_beta_max=float(model_hp["kl_beta_max"]),
            sparse_thresh=float(model_hp["sparse_thresh"]),
            lambda_s=float(model_hp["lambda_s"]),
        )
        model, model_metrics, model_meta = train_dense_vae_plus_sparse(
            X=X,
            d_in=data_cfg.d,
            cont_dim=data_cfg.cont_dim,
            n_sparse=data_cfg.n_sparse,
            cfg=cfg,
            device=DEVICE,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Save model weights and metadata
    torch.save(
        {
            "model_name": model_name,
            "model_class": model.__class__.__name__,
            "state_dict": model.state_dict(),
            "meta": model_meta,
        },
        attempt_dir / "trained_model.pt",
    )

    # Save metrics (make tensors JSON-friendly)
    metrics_jsonable: Dict[str, Any] = {}
    for k, v in model_metrics.items():
        if isinstance(v, torch.Tensor):
            metrics_jsonable[k] = {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "mean": float(v.mean().item()) if v.numel() else None,
                "std": float(v.std().item()) if v.numel() else None,
            }
        else:
            metrics_jsonable[k] = v
    _write_json(attempt_dir / "metrics.json", metrics_jsonable)

    return f"OK {data_cfg.cont_mode} {model_name} seed={seed} -> {attempt_dir}"


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs()
    _write_json(
        RESULTS_ROOT / f"_sweep_manifest_{_now_utc_compact()}.json",
        {"runtime": _runtime_info(), "n_jobs": len(jobs), "jobs": jobs},
    )

    # Decide parallel behavior
    if not ENABLE_PARALLEL:
        max_workers = 1
    elif DEVICE == "cpu":
        max_workers = max(1, int(MAX_WORKERS))
    else:
        # GPU multiprocessing enabled by default; tune MAX_WORKERS_CUDA for your box.
        max_workers = max(1, int(MAX_WORKERS_CUDA))

    if max_workers == 1:
        for job in jobs:
            print(_run_one_job(job), flush=True)
    else:
        # CUDA + multiprocessing on Linux requires 'spawn' (fork will error with
        # "Cannot re-initialize CUDA in forked subprocess").
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        mp_ctx = mp.get_context("spawn") if (DEVICE != "cpu") else mp.get_context()
        n_errors = 0
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as ex:
            futs = [ex.submit(_run_one_job, job) for job in jobs]
            for fut in as_completed(futs):
                try:
                    print(fut.result(), flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"ERROR: {repr(e)}", flush=True)
                    n_errors += 1

        if n_errors:
            raise RuntimeError(f"{n_errors} jobs failed; see ERROR lines above.")

    print(f"Done. Results in: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()

