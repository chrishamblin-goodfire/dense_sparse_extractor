"""
Notebook-style toy generator: embed a data structure into a fixed-size vector.

Goal:
- Build vectors x ∈ R^D as a sum of:
  (1) sparse, symbol-like features (Bernoulli / Top-K / optional tree-structured), and
  (2) dense, continuous features sampled from a low-dim manifold (subspace, sphere, torus, swiss-roll).

Design choices inspired by matryoshka-saes toy model:
- Use (approximately) orthogonal directions for sparse features to avoid geometric confounds,
  then optionally perturb norms slightly to break symmetry.
- Use a separate orthonormal basis for the continuous manifold, and embed into the same D-dim space.
"""

#%%
import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch

# Matplotlib is only used for quick plots.
import matplotlib.pyplot as plt


#%%
"""
Parameters (edit these inline).
"""

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Dimensionality of the unified embedding space.
D = 20

# Dataset size (generated on the fly; nothing is saved to disk).
N = 50_000
BATCH = 2048

# Sparse (symbolic) component
N_SPARSE = 100
SPARSE_MODE: Literal["bernoulli", "topk", "tree_like"] = "bernoulli"
SPARSE_BERNOULLI_P = 0.04
SPARSE_TOPK_K = 3

# Continuous (dense) component
CONT_MODE: Literal[
    "subspace_uniform",
    "subspace_gaussian",
    "ambient_gaussian",
    "sphere",
    "torus",
    "swiss_roll",
    "lowfreq_dct_gaussian",
] = "sphere"
CONT_DIM = 3  # intrinsic latent dimension for subspace/sphere; torus uses 2*n_circles
TORUS_N_CIRCLES = 4  # torus intrinsic dim = 2*TORUS_N_CIRCLES

# Component scaling (how much each contributes to x)
SPARSE_SCALE = 1.0
CONT_SCALE = 0.0

# Optional additional noise in observed x
OBS_NOISE_STD = 0.00

# Direction construction
ORTHOGONALIZE_SPARSE_DIRS = True
ORTHOGONALIZE_CONT_BASIS = True
NORM_JITTER_STD = 0.05  # 0.0 disables; mimics matryoshka-saes toy notebook


#%%
"""
Utilities
"""


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def qr_orthonormal_rows(n_rows: int, n_cols: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Returns an (n_rows x n_cols) matrix with orthonormal rows (when n_rows <= n_cols).
    Constructed by taking QR of a random (n_cols x n_cols) matrix, then taking first n_rows rows.
    """
    if n_rows > n_cols:
        raise ValueError(f"Need n_rows <= n_cols for orthonormal rows, got {n_rows} > {n_cols}.")
    q, _ = torch.linalg.qr(torch.randn(n_cols, n_cols, device=device, dtype=dtype))
    return q[:n_rows]


def maybe_jitter_norms(v: torch.Tensor, std: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multiply each row by (1 + Normal(0, std)) and return (v_scaled, scales).
    """
    if std <= 0:
        scales = torch.ones(v.shape[0], device=v.device, dtype=v.dtype)
        return v, scales
    scales = 1.0 + torch.randn(v.shape[0], device=v.device, dtype=v.dtype) * std
    return v * scales[:, None], scales


def pca_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Simple PCA via SVD. Returns 2D coordinates.
    """
    x0 = x - x.mean(dim=0, keepdim=True)
    # Compute top-2 right singular vectors of centered data
    # (for large N you might want to subsample before calling this)
    _, _, vT = torch.linalg.svd(x0, full_matrices=False)
    return x0 @ vT[:2].T


#%%
"""
Sparse (symbolic) generators
"""


@dataclass
class TreeLikeConfig:
    """
    A lightweight "tree-like" coactivation structure:
    - n_groups groups; each group has a parent + n_children children.
    - parent ~ Bernoulli(p_parent)
    - if parent active: choose one child from categorical child_probs (sum to 1)
      and activate it; optionally include "hidden child" probability mass to produce
      parent-with-no-explicit-child (like matryoshka-saes tree.json).
    """

    n_groups: int = 4
    n_children: int = 3
    p_parent: float = 0.15
    child_probs: Tuple[float, ...] = (0.25, 0.25, 0.25)  # remaining mass -> hidden child
    n_independent_leaves: int = 16
    p_leaf: float = 0.05


def sample_sparse_acts(
    batch: int,
    n_sparse: int,
    mode: Literal["bernoulli", "topk", "tree_like"],
    p: float,
    k: int,
    tree_cfg: Optional[TreeLikeConfig],
    device: str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Returns:
      acts: [batch, n_sparse] nonnegative "symbolic" activations (mostly 0/1)
      meta: helpful extra tensors (e.g., parents/children masks)
    """
    meta: Dict[str, torch.Tensor] = {}

    if mode == "bernoulli":
        acts = (torch.rand(batch, n_sparse, device=device) < p).to(dtype)
        return acts, meta

    if mode == "topk":
        # Choose exactly k active indices per sample (binary).
        if k <= 0 or k > n_sparse:
            raise ValueError(f"Need 1 <= k <= n_sparse; got k={k}, n_sparse={n_sparse}")
        idx = torch.randint(low=0, high=n_sparse, size=(batch, k), device=device)
        acts = torch.zeros(batch, n_sparse, device=device, dtype=dtype)
        acts.scatter_(dim=1, index=idx, value=1.0)
        return acts, meta

    if mode == "tree_like":
        if tree_cfg is None:
            tree_cfg = TreeLikeConfig()

        # Build activations by concatenating:
        # - group parents: [n_groups]
        # - group children: [n_groups * n_children]
        # - independent leaves: [n_independent_leaves]
        n_needed = tree_cfg.n_groups + tree_cfg.n_groups * tree_cfg.n_children + tree_cfg.n_independent_leaves
        if n_needed > n_sparse:
            raise ValueError(f"tree_like needs at least {n_needed} sparse slots, got n_sparse={n_sparse}")

        acts = torch.zeros(batch, n_sparse, device=device, dtype=dtype)

        # Parent slots
        parent_offset = 0
        child_offset = tree_cfg.n_groups
        leaf_offset = child_offset + tree_cfg.n_groups * tree_cfg.n_children

        parents = (torch.rand(batch, tree_cfg.n_groups, device=device) < tree_cfg.p_parent).to(dtype)
        acts[:, parent_offset : parent_offset + tree_cfg.n_groups] = parents

        # Children: mutually exclusive per group when parent is active.
        child_probs = torch.tensor(tree_cfg.child_probs, device=device, dtype=dtype)
        if child_probs.numel() != tree_cfg.n_children:
            raise ValueError("child_probs length must equal n_children")
        if (child_probs < 0).any():
            raise ValueError("child_probs must be nonnegative")
        if float(child_probs.sum()) > 1.0 + 1e-6:
            raise ValueError("child_probs must sum to <= 1.0 (remaining mass is hidden child)")

        # For each group, sample categorical among n_children+1 (hidden) for parent-active samples
        # and force "hidden" when parent-inactive.
        probs_plus_hidden = torch.cat([child_probs, (1.0 - child_probs.sum()).view(1)])
        cat = torch.distributions.Categorical(probs=probs_plus_hidden)

        chosen = cat.sample((batch, tree_cfg.n_groups))  # values in [0..n_children] (n_children == hidden)
        # mask out parent-inactive rows by forcing hidden
        chosen = torch.where(parents.bool(), chosen, torch.full_like(chosen, tree_cfg.n_children))

        children = torch.zeros(batch, tree_cfg.n_groups, tree_cfg.n_children, device=device, dtype=dtype)
        for c in range(tree_cfg.n_children):
            children[:, :, c] = (chosen == c).to(dtype)

        acts[:, child_offset : child_offset + tree_cfg.n_groups * tree_cfg.n_children] = children.reshape(
            batch, -1
        )

        # Independent leaves
        leaves = (torch.rand(batch, tree_cfg.n_independent_leaves, device=device) < tree_cfg.p_leaf).to(dtype)
        acts[:, leaf_offset : leaf_offset + tree_cfg.n_independent_leaves] = leaves

        meta["parents"] = parents
        meta["children"] = children.reshape(batch, -1)
        meta["leaves"] = leaves
        return acts, meta

    raise ValueError(f"Unknown sparse mode: {mode}")


#%%
"""
Continuous (dense) manifold generators
"""


def sample_continuous_latents(
    batch: int,
    mode: Literal[
        "subspace_uniform",
        "subspace_gaussian",
        "ambient_gaussian",
        "sphere",
        "torus",
        "swiss_roll",
        "lowfreq_dct_gaussian",
    ],
    cont_dim: int,
    torus_n_circles: int,
    device: str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Returns:
      z: [batch, z_dim] latent coordinates in an intrinsic space
      meta: extra info (angles, etc.)
    """
    meta: Dict[str, torch.Tensor] = {}

    if mode == "subspace_uniform":
        z = (torch.rand(batch, cont_dim, device=device, dtype=dtype) * 2.0) - 1.0
        return z, meta

    if mode == "subspace_gaussian":
        z = torch.randn(batch, cont_dim, device=device, dtype=dtype)
        return z, meta

    if mode == "ambient_gaussian":
        # Full-dimensional Gaussian in ambient space (we set z_dim = D below).
        # This is the most VAE-friendly dense source.
        # Note: distribution is rotation-invariant; using a random orthonormal basis is fine.
        z = torch.randn(batch, cont_dim, device=device, dtype=dtype)
        return z, meta

    if mode == "sphere":
        # Uniform on the sphere S^{cont_dim-1} (via normalized Gaussian).
        if cont_dim < 2:
            raise ValueError("sphere requires cont_dim >= 2")
        z = torch.randn(batch, cont_dim, device=device, dtype=dtype)
        z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return z, meta

    if mode == "torus":
        # Product of circles; intrinsic dimension = 2*n_circles.
        n = torus_n_circles
        theta = torch.rand(batch, n, device=device, dtype=dtype) * (2 * math.pi)
        # Embed torus in R^{2n}: (cos θ1, sin θ1, ..., cos θn, sin θn)
        z = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1).reshape(batch, 2 * n)
        meta["theta"] = theta
        return z, meta

    if mode == "swiss_roll":
        # Classic 2D swiss roll embedded in 3D; then we can project/rotate into D.
        # Intrinsic: (t, h), where t sets the spiral, h is height.
        # We'll return a 3D embedding as z and treat cont_dim ignored.
        t = (3 * math.pi / 2) * (1 + 2 * torch.rand(batch, 1, device=device, dtype=dtype))
        h = (torch.rand(batch, 1, device=device, dtype=dtype) * 2.0) - 1.0
        x = t * torch.cos(t)
        y = h
        z = t * torch.sin(t)
        pts = torch.cat([x, y, z], dim=1)
        meta["t"] = t.squeeze(1)
        meta["h"] = h.squeeze(1)
        return pts, meta

    if mode == "lowfreq_dct_gaussian":
        # Coefficients for low-frequency DCT basis vectors (smooth 1D signal).
        # This is the most "DenSaE-like" dense source in this repo: it concentrates energy
        # in low spatial frequencies.
        z = torch.randn(batch, cont_dim, device=device, dtype=dtype)
        return z, meta

    raise ValueError(f"Unknown continuous mode: {mode}")


def embed_continuous(
    z: torch.Tensor, basis: torch.Tensor, mode: Literal["swiss_roll", "other"]
) -> torch.Tensor:
    """
    Map intrinsic coordinates z into the ambient D space using a basis.
    basis: [z_dim, D] with orthonormal rows (recommended).
    """
    if z.shape[1] != basis.shape[0]:
        raise ValueError(f"z has dim {z.shape[1]} but basis has {basis.shape[0]} rows")
    return z @ basis


#%%
"""
Build embedding directions / bases for each component.
"""

set_seed(SEED)

# Sparse feature directions: one direction per sparse feature (N_SPARSE).
if ORTHOGONALIZE_SPARSE_DIRS and N_SPARSE <= D:
    sparse_dirs = qr_orthonormal_rows(N_SPARSE, D, device=DEVICE, dtype=DTYPE)
else:
    # Random Gaussian directions, normalized per-row.
    sparse_dirs = torch.randn(N_SPARSE, D, device=DEVICE, dtype=DTYPE)
    sparse_dirs = sparse_dirs / sparse_dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)

sparse_dirs, sparse_scales = maybe_jitter_norms(sparse_dirs, NORM_JITTER_STD)

# Continuous basis: maps z_dim -> D.
#
# Most modes use a random orthonormal basis (a rotated subspace).
# For `lowfreq_dct_gaussian`, we instead use an explicit low-frequency DCT basis,
# which creates a dense component that is smooth (low spatial frequency) and thus
# is amenable to DenSaE-style "smoothness/low-frequency" priors.
if CONT_MODE == "torus":
    z_dim = 2 * TORUS_N_CIRCLES
elif CONT_MODE == "swiss_roll":
    z_dim = 3
elif CONT_MODE == "ambient_gaussian":
    z_dim = D
else:
    z_dim = CONT_DIM


def dct_orthonormal_basis(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Orthonormal DCT-II basis matrix of shape (n, n).
    Treats each vector x ∈ R^n as a 1D "signal"; low k correspond to low spatial frequency.
    """
    nn = torch.arange(n, device=device, dtype=dtype)
    kk = torch.arange(n, device=device, dtype=dtype)[:, None]
    basis = torch.cos(math.pi / n * (nn[None, :] + 0.5) * kk)
    # Orthonormal scaling
    basis[0] *= math.sqrt(1.0 / n)
    if n > 1:
        basis[1:] *= math.sqrt(2.0 / n)
    return basis


if CONT_MODE == "lowfreq_dct_gaussian":
    # First z_dim rows correspond to lowest frequencies.
    cont_basis = dct_orthonormal_basis(D, device=DEVICE, dtype=DTYPE)[:z_dim]
else:
    if ORTHOGONALIZE_CONT_BASIS and z_dim <= D:
        cont_basis = qr_orthonormal_rows(z_dim, D, device=DEVICE, dtype=DTYPE)
    else:
        cont_basis = torch.randn(z_dim, D, device=DEVICE, dtype=DTYPE)
        cont_basis = cont_basis / cont_basis.norm(dim=1, keepdim=True).clamp_min(1e-12)

# Optional: jitter continuous basis row norms too (often set this to 0.0; depends what you want).
cont_basis, cont_scales = maybe_jitter_norms(cont_basis, 0.00)


#%%
"""
Generate a batch / dataset tensors.
"""

tree_cfg = TreeLikeConfig()

all_x = []
all_sparse = []
all_z = []

with torch.no_grad():
    for start in range(0, N, BATCH):
        b = min(BATCH, N - start)

        sparse_acts, sparse_meta = sample_sparse_acts(
            batch=b,
            n_sparse=N_SPARSE,
            mode=SPARSE_MODE,
            p=SPARSE_BERNOULLI_P,
            k=SPARSE_TOPK_K,
            tree_cfg=tree_cfg,
            device=DEVICE,
            dtype=DTYPE,
        )

        z, z_meta = sample_continuous_latents(
            batch=b,
            mode=CONT_MODE,
            cont_dim=CONT_DIM,
            torus_n_circles=TORUS_N_CIRCLES,
            device=DEVICE,
            dtype=DTYPE,
        )

        x_sparse = (sparse_acts @ sparse_dirs) * SPARSE_SCALE
        x_cont = embed_continuous(z, cont_basis, mode="swiss_roll" if CONT_MODE == "swiss_roll" else "other")
        x_cont = x_cont * CONT_SCALE

        x = x_sparse + x_cont
        if OBS_NOISE_STD > 0:
            x = x + torch.randn_like(x) * OBS_NOISE_STD

        all_x.append(x.detach().cpu())
        all_sparse.append(sparse_acts.detach().cpu())
        all_z.append(z.detach().cpu())

X = torch.cat(all_x, dim=0)
S = torch.cat(all_sparse, dim=0)
Z = torch.cat(all_z, dim=0)

print("X:", tuple(X.shape), "S:", tuple(S.shape), "Z:", tuple(Z.shape))
print("X mean/std:", X.mean().item(), X.std().item())
avg_sparse_l0 = (S > 0).float().sum(dim=1).mean().item()
print("avg sparse l0:", avg_sparse_l0)


#%%
"""
Quick sanity checks / visualization.
"""

N_VIZ = 4000
idx = torch.randperm(X.shape[0])[:N_VIZ]
Xv = X[idx]
Sv = S[idx]
Zv = Z[idx]

coords = pca_2d(Xv)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

# Color by a sparse feature (pick one with nontrivial frequency)
freq = Sv.mean(dim=0)
feat_i = int(torch.argmin((freq - freq.median()).abs()).item())
color = Sv[:, feat_i].numpy()
sc = ax[0].scatter(coords[:, 0].numpy(), coords[:, 1].numpy(), c=color, s=6, cmap="viridis")
ax[0].set_title(f"PCA of X (colored by sparse feature {feat_i}, freq={freq[feat_i].item():.3f})")
plt.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.04)

# Scatter color by continuous coordinate magnitude (rough proxy)
cont_mag = Zv.norm(dim=1).numpy()
sc2 = ax[1].scatter(coords[:, 0].numpy(), coords[:, 1].numpy(), c=cont_mag, s=6, cmap="magma")
ax[1].set_title("PCA of X (colored by ||z||)")
plt.colorbar(sc2, ax=ax[1], fraction=0.046, pad=0.04)

# Histogram of per-sample norms
ax[2].hist(Xv.norm(dim=1).numpy(), bins=50, alpha=0.9)
ax[2].set_title("Histogram of ||x||")
ax[2].set_xlabel("||x||")

plt.tight_layout()
plt.show()


#%%
"""
Notes / knobs worth playing with:

- If you want the sparse part to be truly "symbolic", keep sparse_acts binary.
  If you want "sparse continuous" symbols, make acts non-binary (e.g. random positive amplitudes on active indices).

- If you want the manifold to be truly nonlinear in X-space (not just a rotated subspace),
  swiss_roll is a simple classic. You can also create your own f: R^k -> R^D, e.g.:
    z ~ uniform cube
    y = [sin(Az), cos(Bz), z, z^2, ...] @ random_basis
  That gives curvature while keeping sampling controllable.

- If you want sparse and dense to compete for the same dimensions, set D small or
  make sparse_dirs and cont_basis share rows/planes deliberately (not orthogonal).

- If your downstream algorithm is sensitive to norm scaling, try:
    - setting NORM_JITTER_STD = 0.0
    - normalizing X per-sample, or holding SPARSE_SCALE/CONT_SCALE fixed.
"""


#%%
"""
Overcomplete feature extractors

These cells apply Overcomplete's extractors (mostly SAEs) to the combined activations `X`.

Conventions:
- `X` is a torch.Tensor on CPU (created above). We move batches to `OC_DEVICE` during training/encoding.
- Each extractor cell writes results into `OC_FEATURES[name] = codes_cpu`.
- Keep `OC_EPOCHS` small for quick iteration; increase once you're happy with the toy distribution.
"""

from torch.utils.data import DataLoader, TensorDataset

try:
    from overcomplete.sae import (
        SAE,
        TopKSAE,
        BatchTopKSAE,
        JumpSAE,
        RATopKSAE,
        RAJumpSAE,
        QSAE,
        MpSAE,
        OMPSAE,
        train_sae,
        mse_l1,
    )
    # Note: `top_k_auxiliary_loss` is not re-exported at `overcomplete.sae`.
    from overcomplete.sae.losses import top_k_auxiliary_loss
except Exception as e:  # noqa: BLE001
    import sys
    import traceback

    msg = (
        "Failed to import Overcomplete.\n\n"
        f"- sys.executable: {sys.executable}\n"
        f"- sys.path[0]: {sys.path[0] if sys.path else '(empty)'}\n"
        f"- underlying error: {repr(e)}\n\n"
        "Full traceback:\n"
        + "".join(traceback.format_exception(type(e), e, e.__traceback__))
    )
    raise RuntimeError(msg) from e


#%%
"""
Shared training/encoding settings (edit inline).
"""

OC_DEVICE = DEVICE  # usually 'cuda' if available
OC_TRAIN_N = min(20_000, X.shape[0])  # subset for speed
OC_BATCH_SIZE = 500
OC_EPOCHS = 300
OC_LR = 5e-4

# Dictionary width (number of learned features / codes)
OC_N_CONCEPTS = 2*N_SPARSE

# L1 penalty (used by mse_l1); tune to get desired sparsity for Vanilla SAE / JumpSAE / QSAE.
OC_L1_PENALTY = 2e-4

# For JumpSAE, authors recommend standardizing inputs.
OC_STANDARDIZE_X = True

# TopK-related
import math
OC_TOPK = max(1, 2*math.ceil(avg_sparse_l0))
OC_AUXK_PENALTY = 0.1  # only used if you pick top_k_auxiliary_loss


OC_FEATURES = {}  # name -> (N, OC_N_CONCEPTS) tensor on CPU

X_train = X[:OC_TRAIN_N]
X_mean = X_train.mean(dim=0, keepdim=True)
X_std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)

if OC_STANDARDIZE_X:
    X_train_in = (X_train - X_mean) / X_std
    X_full_in = (X - X_mean) / X_std
else:
    X_train_in = X_train
    X_full_in = X

train_loader = DataLoader(
    TensorDataset(X_train_in),
    batch_size=OC_BATCH_SIZE,
    shuffle=True,
    pin_memory=(OC_DEVICE == "cuda"),
)


#%%
def oc_encode_all(model, x_full: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    """
    Encode all samples (CPU tensor) with an Overcomplete SAE, returning codes on CPU.
    """
    model.eval()
    zs = []
    with torch.no_grad():
        for start in range(0, x_full.shape[0], batch_size):
            xb = x_full[start : start + batch_size].to(OC_DEVICE)
            _, z = model.encode(xb)
            zs.append(z.detach().cpu())
    return torch.cat(zs, dim=0)


def oc_train_then_encode(
    name: str,
    model,
    *,
    criterion,
    lr: float = OC_LR,
    epochs: int = OC_EPOCHS,
) -> torch.Tensor:
    """
    Train model on X_train_in, then encode X_full_in. Stores into OC_FEATURES[name].
    """
    model = model.to(OC_DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    _ = train_sae(model, train_loader, criterion, opt, nb_epochs=epochs, device=OC_DEVICE)
    z = oc_encode_all(model, X_full_in)
    OC_FEATURES[name] = z
    print(f"[{name}] codes shape:", tuple(z.shape), "avg L0:", (z > 0).float().sum(dim=1).mean().item())
    return z


#%%
"""
Vanilla SAE (ReLU + learned dictionary)
"""

vanilla = SAE(D, OC_N_CONCEPTS, device=OC_DEVICE)
_ = oc_train_then_encode(
    "vanilla_sae",
    vanilla,
    criterion=lambda x, x_hat, pre, z, d: mse_l1(x, x_hat, pre, z, d, penalty=OC_L1_PENALTY),
)


#%%
"""
TopK SAE
"""

topk = TopKSAE(D, OC_N_CONCEPTS, top_k=OC_TOPK, device=OC_DEVICE)
_ = oc_train_then_encode(
    "topk_sae",
    topk,
    # Option A: plain reconstruction loss (often sufficient for toy data)
    criterion=lambda x, x_hat, pre, z, d: (x - x_hat).square().mean(),
    # Option B: AuxK (uncomment if you want dead-code pressure)
    # criterion=lambda x, x_hat, pre, z, d: top_k_auxiliary_loss(x, x_hat, pre, z, d, penalty=OC_AUXK_PENALTY),
)


#%%
"""
BatchTopK SAE (global top-k over the batch)
"""

batchtopk = BatchTopKSAE(D, OC_N_CONCEPTS, top_k=OC_TOPK, device=OC_DEVICE)
_ = oc_train_then_encode(
    "batchtopk_sae",
    batchtopk,
    criterion=lambda x, x_hat, pre, z, d: (x - x_hat).square().mean(),
)


#%%
"""
JumpSAE (JumpReLU thresholds). Overcomplete notes this is sensitive to input distribution,
so it typically benefits from OC_STANDARDIZE_X=True.
"""

jump = JumpSAE(D, OC_N_CONCEPTS, bandwidth=1e-3, device=OC_DEVICE)
_ = oc_train_then_encode(
    "jump_sae",
    jump,
    criterion=lambda x, x_hat, pre, z, d: mse_l1(x, x_hat, pre, z, d, penalty=OC_L1_PENALTY),
)


#%%
"""
RATopKSAE / RAJumpSAE (reconstruction-aware variants)

These require a `points` argument. For vision, this can be patch positions; for our toy,
we’ll just use per-dimension indices [0..D-1] as a placeholder.
"""

ra_points = torch.arange(D)  # simple placeholder points

ra_topk = RATopKSAE(D, OC_N_CONCEPTS, points=ra_points, top_k=OC_TOPK, device=OC_DEVICE)
_ = oc_train_then_encode(
    "ra_topk_sae",
    ra_topk,
    criterion=lambda x, x_hat, pre, z, d: (x - x_hat).square().mean(),
)


#%%
ra_jump = RAJumpSAE(D, OC_N_CONCEPTS, points=ra_points, bandwidth=1e-3, device=OC_DEVICE)
_ = oc_train_then_encode(
    "ra_jump_sae",
    ra_jump,
    criterion=lambda x, x_hat, pre, z, d: mse_l1(x, x_hat, pre, z, d, penalty=OC_L1_PENALTY),
)


#%%
"""
MpSAE (matching pursuit) — sparse by construction (k steps).
"""

MP_K = 3
mp = MpSAE(D, OC_N_CONCEPTS, k=MP_K, device=OC_DEVICE)
_ = oc_train_then_encode(
    "mp_sae",
    mp,
    criterion=lambda x, x_hat, pre, z, d: (x - x_hat).square().mean(),
)


#%%
"""
OMPSAE (orthogonal matching pursuit + NNLS). Encoding is non-differentiable;
training still updates the dictionary but can be slower/heavier.
"""

OMP_K = 3
omp = OMPSAE(D, OC_N_CONCEPTS, k=OMP_K, device=OC_DEVICE)
_ = oc_train_then_encode(
    "omp_sae",
    omp,
    criterion=lambda x, x_hat, pre, z, d: (x - x_hat).square().mean(),
)


#%%
"""
QSAE (quantized codes). Not sparse by default, but a useful "symbolic-ish" baseline.
"""

qsae = QSAE(D, OC_N_CONCEPTS, q=8, hard=False, device=OC_DEVICE)
_ = oc_train_then_encode(
    "q_sae",
    qsae,
    criterion=lambda x, x_hat, pre, z, d: mse_l1(x, x_hat, pre, z, d, penalty=OC_L1_PENALTY),
)


#%%
"""
(Optional) Optimization-based extractors (sklearn wrappers + NMF family).

These require scikit-learn for the sklearn wrappers. NMF methods require nonnegative X.
"""

try:
    from overcomplete.optimization import (
        SkPCA,
        SkICA,
        SkSVD,
        SkKMeans,
        SkDictionaryLearning,
        SkSparsePCA,
        NMF as TorchNMF,
        SemiNMF,
        ConvexNMF,
    )

    HAVE_OPT = True
except Exception as e:  # noqa: BLE001
    HAVE_OPT = False
    print("Optimization extractors unavailable (missing optional deps?). Error:", repr(e))


#%%
# PCA (sklearn wrapper)
if HAVE_OPT:
    pca = SkPCA(nb_concepts=min(D, 16), device="cpu")
    pca.fit(X_train.numpy())
    Z_pca = pca.encode(X.numpy()).cpu()
    OC_FEATURES["sk_pca"] = Z_pca
    print("[sk_pca] codes shape:", tuple(Z_pca.shape))


#%%
# ICA (sklearn wrapper)
if HAVE_OPT:
    ica = SkICA(nb_concepts=min(D, 16), device="cpu")
    ica.fit(X_train.numpy())
    Z_ica = ica.encode(X.numpy()).cpu()
    OC_FEATURES["sk_ica"] = Z_ica
    print("[sk_ica] codes shape:", tuple(Z_ica.shape))


#%%
# DictionaryLearning (sklearn wrapper; sparse codes)
if HAVE_OPT:
    dl = SkDictionaryLearning(nb_concepts=OC_N_CONCEPTS, device="cpu", sparsity=1.0)
    dl.fit(X_train.numpy())
    Z_dl = dl.encode(X.numpy()).cpu()
    OC_FEATURES["sk_dict_learning"] = Z_dl
    print("[sk_dict_learning] codes shape:", tuple(Z_dl.shape), "avg L0:", (Z_dl != 0).float().sum(dim=1).mean().item())


#%%
# SparsePCA (sklearn wrapper)
if HAVE_OPT:
    spca = SkSparsePCA(nb_concepts=min(D, 16), device="cpu", sparsity=1.0)
    spca.fit(X_train.numpy())
    Z_spca = spca.encode(X.numpy()).cpu()
    OC_FEATURES["sk_sparse_pca"] = Z_spca
    print("[sk_sparse_pca] codes shape:", tuple(Z_spca.shape))


#%%
# TruncatedSVD (sklearn wrapper)
if HAVE_OPT:
    svd = SkSVD(nb_concepts=min(D, 16), device="cpu")
    svd.fit(X_train.numpy())
    Z_svd = svd.encode(X.numpy()).cpu()
    OC_FEATURES["sk_svd"] = Z_svd
    print("[sk_svd] codes shape:", tuple(Z_svd.shape))


#%%
# KMeans (sklearn wrapper)
if HAVE_OPT:
    km = SkKMeans(nb_concepts=16, device="cpu")
    km.fit(X_train.numpy())
    Z_km = km.encode(X.numpy()).cpu()
    OC_FEATURES["sk_kmeans"] = Z_km
    print("[sk_kmeans] codes shape:", tuple(Z_km.shape))


#%%
# NMF family (requires nonnegative data)
if HAVE_OPT:
    X_nonneg = X - X.min()
    nmf = TorchNMF(nb_concepts=min(64, max(8, D)), device=OC_DEVICE, solver="hals")
    nmf.fit(X_nonneg.to(OC_DEVICE), max_iter=300)
    Z_nmf = nmf.encode(X_nonneg.to(OC_DEVICE), max_iter=100).cpu()
    OC_FEATURES["nmf"] = Z_nmf
    print("[nmf] codes shape:", tuple(Z_nmf.shape))


#%%
"""
Recovering the *true sparse activations* S from learned SAE features / dictionaries.

We know two kinds of ground truth:
- `S`: per-sample sparse activations (shape [N, N_SPARSE])
- `sparse_dirs`: the true directions in X-space for each sparse feature (shape [N_SPARSE, D])

Two complementary metrics:
1) **Dictionary-direction recovery** (geometry-only):
   For each true sparse direction v_i, compute max cosine similarity with any learned atom d_j.
   High values mean the learned dictionary contains vectors aligned to the true sparse directions.

2) **S recovery via codes** (activation-level):
   Align learned atoms to true sparse directions (Hungarian matching), then treat the matched code z_j
   as a predictor for S_i and report AUROC / Average Precision.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score


#%%
def _normalize_rows(a: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return a / a.norm(dim=1, keepdim=True).clamp_min(eps)


def dict_sims_to_sparse_dirs(model) -> torch.Tensor:
    """
    Returns cosine similarity matrix sims[j, i] between learned dict atom j and true sparse dir i.
    Shape: [nb_concepts, N_SPARSE]
    """
    # Overcomplete dictionaries are returned as (nb_concepts, D)
    d_learned = model.get_dictionary().detach().cpu()
    v_true = sparse_dirs.detach().cpu()
    sims = _normalize_rows(d_learned) @ _normalize_rows(v_true).T
    return sims


def greedy_best_cosine_per_true_feature(sims: torch.Tensor) -> torch.Tensor:
    """
    For each true feature i, return max_j sims[j, i].
    """
    return sims.max(dim=0).values


def hungarian_match(sims: torch.Tensor) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    One-to-one matching between learned atoms (rows) and true sparse dirs (cols).
    Returns (row_ind, col_ind, matched_sims) where matched_sims[k] = sims[row_ind[k], col_ind[k]].
    """
    sims_np = sims.numpy()
    # Convert to cost for minimization. We want maximize sims -> minimize (max - sims).
    maxv = float(np.max(sims_np))
    cost = maxv - sims_np
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = sims[row_ind, col_ind]
    return row_ind, col_ind, matched


def s_recovery_metrics_from_codes(
    z: torch.Tensor,
    s_true: torch.Tensor,
    *,
    row_ind: np.ndarray,
    col_ind: np.ndarray,
) -> dict:
    """
    Evaluate AUROC/AP for predicting each true S_i from its matched code z_j.

    z: [N, n_latents] (CPU)
    s_true: [N, N_SPARSE] (CPU, binary/float)
    row_ind, col_ind: arrays from hungarian_match where z[:, row_ind[k]] is aligned to S[:, col_ind[k]]
    """
    z_np = z.numpy()
    s_np = s_true.numpy()

    aucs = []
    aps = []

    for j, i in zip(row_ind, col_ind):
        y = s_np[:, i]
        # skip degenerate labels
        if y.min() == y.max():
            continue
        scores = z_np[:, j]
        # AUROC can fail if all scores are equal; handle robustly
        try:
            aucs.append(roc_auc_score(y, scores))
        except Exception:  # noqa: BLE001
            pass
        try:
            aps.append(average_precision_score(y, scores))
        except Exception:  # noqa: BLE001
            pass

    out = {
        "mean_auroc": float(np.mean(aucs)) if len(aucs) else float("nan"),
        "mean_ap": float(np.mean(aps)) if len(aps) else float("nan"),
        "n_eval": int(max(len(aucs), len(aps))),
    }
    return out


#%%
"""
Pick which trained SAE models to evaluate.

If you ran the SAE training cells above, you should have variables like `vanilla`, `topk`, etc.
"""

OC_MODELS = {
    # SAE family
    "vanilla_sae": globals().get("vanilla", None),
    "topk_sae": globals().get("topk", None),
    "batchtopk_sae": globals().get("batchtopk", None),
    "jump_sae": globals().get("jump", None),
    "ra_topk_sae": globals().get("ra_topk", None),
    "ra_jump_sae": globals().get("ra_jump", None),
    "mp_sae": globals().get("mp", None),
    "omp_sae": globals().get("omp", None),
    "q_sae": globals().get("qsae", None),
}

# Drop ones you didn't run
OC_MODELS = {k: v for k, v in OC_MODELS.items() if v is not None}
print("Models found for evaluation:", list(OC_MODELS.keys()))


#%%
"""
Compute and summarize dictionary + S recovery metrics.
"""

S_cpu = S.detach().cpu().float()
results = {}
alignments = {}  # model_name -> dict with matching info

for name, model in OC_MODELS.items():
    sims = dict_sims_to_sparse_dirs(model)  # [n_latents, N_SPARSE]
    best_cos = greedy_best_cosine_per_true_feature(sims)  # [N_SPARSE]

    row_ind, col_ind, matched = hungarian_match(sims)
    # Build a true-feature -> latent index mapping (size N_SPARSE, -1 if unmapped)
    mapping = -np.ones(N_SPARSE, dtype=int)
    mapping[col_ind] = row_ind
    alignments[name] = {
        "row_ind": row_ind,
        "col_ind": col_ind,
        "mapping_true_to_latent": mapping,
        "matched_cos": matched.detach().cpu(),
    }

    # Use codes if available; otherwise encode once.
    if name in OC_FEATURES:
        z = OC_FEATURES[name]
    else:
        z = oc_encode_all(model, X_full_in)
        OC_FEATURES[name] = z

    s_metrics = s_recovery_metrics_from_codes(z, S_cpu, row_ind=row_ind, col_ind=col_ind)

    results[name] = {
        "mean_best_cos": float(best_cos.mean().item()),
        "median_best_cos": float(best_cos.median().item()),
        "mean_matched_cos": float(matched.mean().item()),
        "median_matched_cos": float(matched.median().item()),
        **s_metrics,
    }

print("==== Sparse recovery summary ====")
for name, r in results.items():
    print(
        f"{name:14s} | best cos mean={r['mean_best_cos']:.3f} (med {r['median_best_cos']:.3f}) "
        f"| matched cos mean={r['mean_matched_cos']:.3f} "
        f"| AUROC={r['mean_auroc']:.3f} AP={r['mean_ap']:.3f} (n={r['n_eval']})"
    )


#%%
"""
Plots:
- Heatmap of dictionary-vs-true sparse direction cosine similarities for a chosen model.
- Histogram of per-true-feature best cosine similarity (coverage).
"""

MODEL_TO_PLOT = next(iter(results.keys())) if len(results) else None
N_TRUE_PLOT = min(N_SPARSE, 40)
N_LATENT_PLOT = min(OC_N_CONCEPTS, 60)

if MODEL_TO_PLOT is None:
    print("No models available to plot.")
else:
    model = OC_MODELS[MODEL_TO_PLOT]
    sims = dict_sims_to_sparse_dirs(model)[:N_LATENT_PLOT, :N_TRUE_PLOT]
    best_cos = greedy_best_cosine_per_true_feature(dict_sims_to_sparse_dirs(model))

    plt.figure(figsize=(10, 4))
    plt.imshow(sims.numpy(), aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar()
    plt.title(f"Cosine sim: learned dict vs true sparse dirs ({MODEL_TO_PLOT})")
    plt.xlabel("true sparse feature i")
    plt.ylabel("learned atom j")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 3))
    plt.hist(best_cos.numpy(), bins=40, alpha=0.9)
    plt.title(f"Best cosine per true sparse feature ({MODEL_TO_PLOT})")
    plt.xlabel("max_j cos(d_j, v_i)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


#%%
"""
Per-feature diagnostic plots:
For selected true sparse features i, plot distributions of the *matched* latent z_j
conditioned on S_i ∈ {0,1}. This helps you see whether the encoder codes actually
track the true on/off variable (even when the dictionary direction is good).
"""

# Choose models/features to diagnose
MODELS_TO_DIAG = [k for k in ["vanilla_sae", "topk_sae", "mp_sae"] if k in OC_MODELS]
N_FEATURES_TO_DIAG = 6

# Pick "non-degenerate" sparse features: avoid too-rare/too-common
freq = S_cpu.mean(dim=0).numpy()
candidate = np.where((freq > 0.01) & (freq < 0.30))[0]
if len(candidate) == 0:
    candidate = np.argsort(np.abs(freq - np.median(freq)))  # fallback
FEATURES_TO_DIAG = candidate[:N_FEATURES_TO_DIAG].tolist()

print("Models:", MODELS_TO_DIAG)
print("Features:", FEATURES_TO_DIAG)
print("Feature freqs:", [float(freq[i]) for i in FEATURES_TO_DIAG])

for i in FEATURES_TO_DIAG:
    y = S_cpu[:, i].numpy()
    if y.min() == y.max():
        continue

    nrows = len(MODELS_TO_DIAG)
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 2.6 * nrows), sharex=False)
    if nrows == 1:
        axes = [axes]

    for ax, model_name in zip(axes, MODELS_TO_DIAG):
        z = OC_FEATURES[model_name].numpy()
        j = int(alignments[model_name]["mapping_true_to_latent"][i])
        if j < 0:
            ax.set_title(f"{model_name}: no matched latent for feature {i}")
            continue

        scores = z[:, j]
        s0 = scores[y == 0]
        s1 = scores[y == 1]

        # Compute per-feature metrics
        try:
            auc = roc_auc_score(y, scores)
        except Exception:  # noqa: BLE001
            auc = float("nan")
        try:
            ap = average_precision_score(y, scores)
        except Exception:  # noqa: BLE001
            ap = float("nan")

        ax.hist(s0, bins=60, alpha=0.6, density=True, label=f"S=0 (n={len(s0)})")
        ax.hist(s1, bins=60, alpha=0.6, density=True, label=f"S=1 (n={len(s1)})")
        ax.set_title(
            f"true feature i={i} (freq={freq[i]:.3f})  |  {model_name}: latent j={j}  "
            f"|  AUROC={auc:.3f}  AP={ap:.3f}"
        )
        ax.set_xlabel("matched latent activation z_j")
        ax.set_ylabel("density")
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


#%%
"""
DenSaE-inspired two-branch model for this toy

Your data is: X = (sparse part) + (dense part).
Standard SAEs assume "everything should be sparse", which can make sparse codes absorb dense variation.

DenSaE's key idea (adapted to vectors):
- Maintain TWO codes:
    z_t : dense code (no shrinkage)
    s_t : sparse code (thresholded / shrinkage)
- Reconstruct with TWO dictionaries:
    x_hat = z_T @ A + s_T @ B
- Infer (z_T, s_T) by unrolling a few proximal-gradient steps, where only the sparse stream
  gets the shrinkage nonlinearity.

This is the simplest "MCA / DenSaE" baseline you can deploy like an SAE:
encode -> (z, s), decode -> x_hat, and you can ablate s or z separately.
"""

import torch.nn as nn
import torch.nn.functional as F


#%%
def soft_shrink(x: torch.Tensor, thresh: float, twosided: bool) -> torch.Tensor:
    if twosided:
        # classic soft thresholding (L1 prox)
        return torch.sign(x) * torch.relu(torch.abs(x) - thresh)
    # nonnegative shrinkage (ReLU_b), good match for your binary S toy
    return torch.relu(x - thresh)


class ToyDenSaE(nn.Module):
    """
    Vector-valued DenSaE-like unrolled proximal model:
      minimize 0.5||x - zA - sB||^2 + (1/(2*lambda_x))||zA||^2 + lam*||s||_1  (approx)

    - Dense branch uses gradient steps (no shrinkage).
    - Sparse branch uses gradient steps + shrinkage.
    - A and B are learned dictionaries (row-normalized).
    """

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
        inv_lambda_x: float = 0.0,  # DenSaE-style dense Tikhonov strength: (1/lambda_x)*||zA||^2
        device: str = "cpu",
    ):
        super().__init__()
        self.d_in = d_in
        self.d_dense = d_dense
        self.d_sparse = d_sparse
        self.n_iters = n_iters
        self.alpha_z = alpha_z
        self.alpha_s = alpha_s
        self.thresh = thresh
        self.twosided = twosided
        self.inv_lambda_x = inv_lambda_x
        self.device = device

        # Dictionaries as decoder matrices (rows are atoms in data space)
        # A: [d_dense, d_in], B: [d_sparse, d_in]
        A0 = torch.randn(d_dense, d_in, device=device)
        B0 = torch.randn(d_sparse, d_in, device=device)

        # Row-normalize to reduce scaling pathologies; training will renormalize too.
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
        # Initialize codes at 0.
        z = torch.zeros(x.shape[0], self.d_dense, device=x.device, dtype=x.dtype)
        s = torch.zeros(x.shape[0], self.d_sparse, device=x.device, dtype=x.dtype)

        for _ in range(self.n_iters):
            x_hat, x_dense, _ = self.decode(z, s)
            res = x - x_hat

            # Dense update: gradient step on recon + optional ||x_dense||^2 penalty
            # grad_z of 0.5||res||^2 is -res @ A^T
            # so GD ascent form: z += alpha * (res @ A^T)
            z = z + self.alpha_z * (res @ self.A.T)
            if self.inv_lambda_x > 0:
                # DenSaE-style dense Tikhonov: (1/(2*lambda_x))||x_dense||^2
                # grad wrt z is (1/lambda_x) * (x_dense @ A^T)
                z = z - self.alpha_z * (self.inv_lambda_x * (x_dense @ self.A.T))

            # Sparse update: gradient step + shrinkage
            s_pre = s + self.alpha_s * (res @ self.B.T)
            s = soft_shrink(s_pre, self.thresh, self.twosided)

        return z, s

    def forward(self, x: torch.Tensor):
        z, s = self.encode(x)
        x_hat, x_dense, x_sparse = self.decode(z, s)
        return x_hat, z, s, x_dense, x_sparse


#%%
"""
Train the ToyDenSaE model on X (no saving, notebook-style).
"""

DS_DEVICE = DEVICE

# Choose capacities. If you don't know true dims, make them "big enough" and rely on penalties/sparsity.
DS_DENSE = max(2, CONT_DIM)  # heuristic default: try matching toy intrinsic dim
DS_SPARSE = OC_N_CONCEPTS if "OC_N_CONCEPTS" in globals() else max(4 * D, 64)

DS_ITERS = 15
DS_ALPHA_Z = 0.25
DS_ALPHA_S = 0.25
DS_THRESH = 0.10  # increase for more sparsity
DS_TWOSIDED = False

# Dense regularization knobs (DenSaE-style)
# - `DS_INV_LAMBDA_X_UPDATE` affects the *inference updates* (closest to DenSaE / unrolling).
# - `DS_DENSE_REG_WEIGHT_LOSS` adds an explicit penalty in the *training loss*.
DS_INV_LAMBDA_X_UPDATE = 0.0   # set e.g. 0.1, 1.0, 3.0 to test dense Tikhonov pressure in the update
DS_DENSE_REG_WEIGHT_LOSS = 0.0 # set e.g. 1e-3, 1e-2 if you want extra dense penalty in loss

DS_LR = 3e-3
DS_EPOCHS = 50
DS_BATCH = 1024

# Input standardization generally helps these unrolled schemes.
DS_STANDARDIZE_X = True

X_ds_train = X[: min(20_000, X.shape[0])].to(DS_DEVICE)
ds_mean = X_ds_train.mean(dim=0, keepdim=True)
ds_std = X_ds_train.std(dim=0, keepdim=True).clamp_min(1e-6)

if DS_STANDARDIZE_X:
    X_ds_train_in = (X_ds_train - ds_mean) / ds_std
    X_ds_full_in = ((X.to(DS_DEVICE) - ds_mean) / ds_std)
else:
    X_ds_train_in = X_ds_train
    X_ds_full_in = X.to(DS_DEVICE)

ds_loader = DataLoader(
    TensorDataset(X_ds_train_in.detach().cpu()),
    batch_size=DS_BATCH,
    shuffle=True,
    pin_memory=(DS_DEVICE == "cuda"),
)

toy_densae = ToyDenSaE(
    d_in=D,
    d_dense=DS_DENSE,
    d_sparse=DS_SPARSE,
    n_iters=DS_ITERS,
    alpha_z=DS_ALPHA_Z,
    alpha_s=DS_ALPHA_S,
    thresh=DS_THRESH,
    twosided=DS_TWOSIDED,
    inv_lambda_x=DS_INV_LAMBDA_X_UPDATE,
    device=DS_DEVICE,
).to(DS_DEVICE)

opt = torch.optim.Adam(toy_densae.parameters(), lr=DS_LR)

for epoch in range(DS_EPOCHS):
    toy_densae.train()
    total = 0.0
    for (xb_cpu,) in ds_loader:
        xb = xb_cpu.to(DS_DEVICE, non_blocking=True)
        x_hat, z, s, x_dense, x_sparse = toy_densae(xb)

        recon = (xb - x_hat).square().mean()
        # Encourage sparsity in s; if twosided=False, this is just mean(s).
        sparse_pen = s.abs().mean()
        # Optional dense Tikhonov penalty on x_dense energy (DenSaE-style bias in data space)
        dense_pen = x_dense.square().mean()

        # Note: DS_THRESH is the *prox threshold*; this L1 weight is an additional training pressure.
        loss = recon + (1e-3 * sparse_pen) + (DS_DENSE_REG_WEIGHT_LOSS * dense_pen)

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            toy_densae.normalize_dicts_()
            total += float(loss.item())

    if (epoch + 1) % max(1, DS_EPOCHS // 10) == 0 or epoch == 0:
        toy_densae.eval()
        with torch.no_grad():
            x_hat, z, s, *_ = toy_densae(X_ds_train_in[:2000].to(DS_DEVICE))
            l0 = (s.abs() > 1e-8).float().sum(dim=1).mean().item()
            print(f"[ToyDenSaE] epoch {epoch+1:4d}/{DS_EPOCHS}  loss={total/len(ds_loader):.4f}  recon={((X_ds_train_in[:2000].to(DS_DEVICE)-x_hat).square().mean()).item():.4f}  s_L0={l0:.2f}")


#%%
"""
Encode full dataset and evaluate S-recovery using the *s branch*.

We reuse the AUROC/AP logic, but since ToyDenSaE has two dictionaries (A and B),
we align the *sparse dictionary* B to the true sparse directions.
"""

toy_densae.eval()
with torch.no_grad():
    # Full encode (might be a bit heavy; reduce N if needed)
    z_all = []
    s_all = []
    for start in range(0, X_ds_full_in.shape[0], 4096):
        xb = X_ds_full_in[start : start + 4096]
        _, z, s, *_ = toy_densae(xb)
        z_all.append(z.detach().cpu())
        s_all.append(s.detach().cpu())

Z_DS = torch.cat(z_all, dim=0)
S_DS = torch.cat(s_all, dim=0)  # inferred sparse codes
print("ToyDenSaE codes:", tuple(Z_DS.shape), tuple(S_DS.shape))

# Align learned B atoms to true sparse dirs
with torch.no_grad():
    B_learned = toy_densae.B.detach().cpu()  # [DS_SPARSE, D]
    V_true = sparse_dirs.detach().cpu()      # [N_SPARSE, D]
    sims_B = _normalize_rows(B_learned) @ _normalize_rows(V_true).T  # [DS_SPARSE, N_SPARSE]

best_cos = sims_B.max(dim=0).values
row_ind, col_ind, matched = hungarian_match(sims_B)

metrics = s_recovery_metrics_from_codes(S_DS, S_cpu, row_ind=row_ind, col_ind=col_ind)
print(
    "[ToyDenSaE] best cos mean={:.3f} (med {:.3f}) | matched cos mean={:.3f} | AUROC={:.3f} AP={:.3f} (n={})".format(
        float(best_cos.mean().item()),
        float(best_cos.median().item()),
        float(matched.mean().item()),
        metrics["mean_auroc"],
        metrics["mean_ap"],
        metrics["n_eval"],
    )
)


#%%
"""
Additive Dense+Sparse model where the dense component is a VAE

Goal: x ≈ x_dense(z) + x_sparse(s)
- Dense: VAE with Gaussian latent z and MLP decoder producing x_dense ∈ R^D
- Sparse: thresholded code s (ReLU_b or soft-shrink) with linear dictionary B producing x_sparse = s @ B

Loss (typical):
  recon(x, x_hat) + beta * KL(q(z|x) || N(0, I)) + lambda_s * ||s||_1

This is "DenSaE in spirit" (two independently-regularized representations that add),
but uses a VAE-style dense component rather than a Tikhonov/smoothness prior.
"""


#%%
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
        # KL scheduling (baked into the model as a helper; training loop supplies global step)
        kl_beta_max: float = 1e-3,
        kl_warmup_steps: int = 0,
        kl_ramp_steps: int = 10_000,
        kl_schedule: Literal["constant", "linear"] = "linear",
        device: str = "cpu",
    ):
        super().__init__()
        self.d_in = d_in
        self.z_dim = z_dim
        self.d_sparse = d_sparse
        self.twosided_sparse = twosided_sparse
        self.sparse_thresh = sparse_thresh
        self.device = device

        self.kl_beta_max = float(kl_beta_max)
        self.kl_warmup_steps = int(kl_warmup_steps)
        self.kl_ramp_steps = int(kl_ramp_steps)
        self.kl_schedule = kl_schedule

        # VAE encoder q(z|x)
        self.enc = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden, z_dim)
        self.enc_logvar = nn.Linear(hidden, z_dim)

        # VAE decoder p(x_dense|z) (deterministic mean reconstruction)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_in),
        )

        # Sparse encoder: amortized codes from x (then threshold)
        self.sparse_enc = nn.Linear(d_in, d_sparse, bias=True)

        # Sparse dictionary (decoder): B is [d_sparse, d_in]
        B0 = torch.randn(d_sparse, d_in, device=device)
        B0 = B0 / B0.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.B = nn.Parameter(B0)

    def kl_beta(self, step: int) -> float:
        """
        Return the KL weight β at a given global step.
        Common practice: warmup then ramp β from 0 → β_max.
        """
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
    # KL(q||p) for q=N(mu, diag(exp(logvar))) and p=N(0, I)
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1).mean()


#%%
"""
Train DenseVAEPlusSparse on X.

Note: Gaussian-latent VAE is most naturally matched to CONT_MODE="ambient_gaussian".
For sphere/torus-like dense sources, KL may fight the geometry; you can set BETA_KL small.
"""

VS_DEVICE = DEVICE

VS_ZDIM = max(2, CONT_DIM)         # VAE latent dim
VS_SPARSE = DS_SPARSE if "DS_SPARSE" in globals() else max(4 * D, 64)
VS_HIDDEN = 128

VS_SPARSE_THRESH = 0.10
VS_TWOSIDED = False

VS_LR = 3e-3
VS_EPOCHS = 50
VS_BATCH = 1024

VS_LAMBDA_S = 1e-3   # sparsity regularizer strength on s

# KL scheduling (common in VAE training)
VS_KL_BETA_MAX = 1e-3     # final β
VS_KL_WARMUP_STEPS = 0    # keep β=0 for first steps
VS_KL_RAMP_STEPS = 10_000 # then linearly ramp β to β_max over this many optimizer steps
VS_KL_SCHEDULE: Literal["constant", "linear"] = "linear"

VS_STANDARDIZE_X = True

X_vs_train = X[: min(20_000, X.shape[0])].to(VS_DEVICE)
vs_mean = X_vs_train.mean(dim=0, keepdim=True)
vs_std = X_vs_train.std(dim=0, keepdim=True).clamp_min(1e-6)

if VS_STANDARDIZE_X:
    X_vs_train_in = (X_vs_train - vs_mean) / vs_std
    X_vs_full_in = ((X.to(VS_DEVICE) - vs_mean) / vs_std)
else:
    X_vs_train_in = X_vs_train
    X_vs_full_in = X.to(VS_DEVICE)

vs_loader = DataLoader(
    TensorDataset(X_vs_train_in.detach().cpu()),
    batch_size=VS_BATCH,
    shuffle=True,
    pin_memory=(VS_DEVICE == "cuda"),
)

vae_sparse = DenseVAEPlusSparse(
    d_in=D,
    z_dim=VS_ZDIM,
    d_sparse=VS_SPARSE,
    hidden=VS_HIDDEN,
    twosided_sparse=VS_TWOSIDED,
    sparse_thresh=VS_SPARSE_THRESH,
    kl_beta_max=VS_KL_BETA_MAX,
    kl_warmup_steps=VS_KL_WARMUP_STEPS,
    kl_ramp_steps=VS_KL_RAMP_STEPS,
    kl_schedule=VS_KL_SCHEDULE,
    device=VS_DEVICE,
).to(VS_DEVICE)

opt = torch.optim.Adam(vae_sparse.parameters(), lr=VS_LR)

global_step = 0
for epoch in range(VS_EPOCHS):
    vae_sparse.train()
    total = 0.0
    for (xb_cpu,) in vs_loader:
        xb = xb_cpu.to(VS_DEVICE, non_blocking=True)
        x_hat, (mu, logvar, _z), (_pre_s, s), (_xd, _xs) = vae_sparse(xb)

        recon = (xb - x_hat).square().mean()
        kl = kl_diag_gaussian(mu, logvar)
        sparse_pen = s.abs().mean()
        beta = vae_sparse.kl_beta(global_step)
        loss = recon + (beta * kl) + (VS_LAMBDA_S * sparse_pen)

        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            vae_sparse.normalize_dict_()
            total += float(loss.item())
        global_step += 1

    if (epoch + 1) % max(1, VS_EPOCHS // 10) == 0 or epoch == 0:
        vae_sparse.eval()
        with torch.no_grad():
            x_hat, (mu, logvar, _), (_, s), _ = vae_sparse(X_vs_train_in[:2000].to(VS_DEVICE))
            l0 = (s.abs() > 1e-8).float().sum(dim=1).mean().item()
            print(
                f"[VAE+Sparse] epoch {epoch+1:4d}/{VS_EPOCHS} "
                f"loss={total/len(vs_loader):.4f} recon={((X_vs_train_in[:2000].to(VS_DEVICE)-x_hat).square().mean()).item():.4f} "
                f"KL={kl_diag_gaussian(mu, logvar).item():.4f} beta={vae_sparse.kl_beta(global_step):.2e} s_L0={l0:.2f}"
            )


#%%
"""
Evaluate S recovery from the sparse branch of DenseVAEPlusSparse.
"""

vae_sparse.eval()
with torch.no_grad():
    S_VS_list = []
    Z_VS_mu_list = []
    for start in range(0, X_vs_full_in.shape[0], 4096):
        xb = X_vs_full_in[start : start + 4096]
        _xhat, (mu, _logvar, _z), (_pre_s, s), _ = vae_sparse(xb)
        S_VS_list.append(s.detach().cpu())
        Z_VS_mu_list.append(mu.detach().cpu())

S_VS = torch.cat(S_VS_list, dim=0)
Z_VS = torch.cat(Z_VS_mu_list, dim=0)  # use mu as dense representation
print("VAE+Sparse codes:", tuple(Z_VS.shape), tuple(S_VS.shape))

with torch.no_grad():
    B_learned = vae_sparse.B.detach().cpu()  # [VS_SPARSE, D]
    V_true = sparse_dirs.detach().cpu()      # [N_SPARSE, D]
    sims_B = _normalize_rows(B_learned) @ _normalize_rows(V_true).T

best_cos = sims_B.max(dim=0).values
row_ind, col_ind, matched = hungarian_match(sims_B)
metrics = s_recovery_metrics_from_codes(S_VS, S_cpu, row_ind=row_ind, col_ind=col_ind)

print(
    "[VAE+Sparse] best cos mean={:.3f} (med {:.3f}) | matched cos mean={:.3f} | AUROC={:.3f} AP={:.3f} (n={})".format(
        float(best_cos.mean().item()),
        float(best_cos.median().item()),
        float(matched.mean().item()),
        metrics["mean_auroc"],
        metrics["mean_ap"],
        metrics["n_eval"],
    )
)


#%%
