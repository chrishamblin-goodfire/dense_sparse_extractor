from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_threshold(x: torch.Tensor, thr: float, *, twosided: bool) -> torch.Tensor:
    t = float(thr)
    if t <= 0:
        return x
    if twosided:
        return torch.sign(x) * F.relu(x.abs() - t)
    return F.relu(x - t)


@dataclass(frozen=True)
class DenSAEConfig:
    input_dim: int = 28 * 28
    n_dense: int = 256
    n_sparse: int = 2048
    n_iters: int = 10
    # Prox-grad step sizes (can be tuned; smaller is more stable).
    step_x: float = 1e-2
    step_u: float = 1e-2
    # Regularization strengths for the unrolled objective:
    #   0.5||y - Ax - Bu||^2 + 0.5*lambda_x*||x||^2 + lambda_u*||u||_1
    lambda_x: float = 0.0
    lambda_u: float = 1e-2
    fista: bool = True
    twosided_u: bool = True
    # Column-normalize A,B at init to reduce scale issues.
    normalize_init: bool = True


class DenSAE(nn.Module):
    """
    Dense-and-Sparse Autoencoder (DenSaE) for vector inputs.

    This is a small, self-contained PyTorch adaptation of the DenSaE idea from the paper:
    unroll T proximal-gradient (optionally FISTA) iterations to infer a dense code x and a
    sparse code u under y ≈ A x + B u, then decode via the same generative model.

    Output format matches Overcomplete SAEs: (pre_codes, codes, x_hat),
    where codes are the sparse codes u (after shrinkage), and x_hat are reconstruction logits.
    """

    def __init__(self, cfg: DenSAEConfig, *, device: torch.device | None = None):
        super().__init__()
        self.cfg = cfg
        self.input_dim = int(cfg.input_dim)
        self.n_dense = int(cfg.n_dense)
        self.n_sparse = int(cfg.n_sparse)
        self.n_iters = int(cfg.n_iters)

        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.n_dense <= 0:
            raise ValueError("n_dense must be > 0")
        if self.n_sparse <= 0:
            raise ValueError("n_sparse must be > 0")
        if self.n_iters <= 0:
            raise ValueError("n_iters must be > 0")

        d = self.input_dim
        # Dictionaries: shape (D, K) so decode is y_hat = x @ A.T + u @ B.T.
        A = torch.randn(d, self.n_dense)
        B = torch.randn(d, self.n_sparse)
        if bool(cfg.normalize_init):
            A = F.normalize(A, dim=0)
            B = F.normalize(B, dim=0)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)

        if device is not None:
            self.to(device)

    @torch.no_grad()
    def normalize(self) -> None:
        """Column-normalize A and B (useful occasionally during training)."""
        self.A.data = F.normalize(self.A.data, dim=0)
        self.B.data = F.normalize(self.B.data, dim=0)

    def decode(self, *, x_dense: torch.Tensor, u_sparse: torch.Tensor) -> torch.Tensor:
        # (B, Kx) @ (Kx, D) + (B, Ku) @ (Ku, D) -> (B, D)
        return x_dense @ self.A.t() + u_sparse @ self.B.t()

    def forward(self, y: torch.Tensor):
        """
        Parameters
        ----------
        y : torch.Tensor
            Input tensor of shape (batch, D).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (u_pre, u, y_hat_logits) where:
            - u_pre: pre-threshold sparse codes (batch, n_sparse)
            - u: thresholded sparse codes (batch, n_sparse)
            - y_hat_logits: reconstructed logits (batch, D)
        """
        if y.ndim != 2:
            raise ValueError(f"DenSAE expects 2D input (B,D), got shape {tuple(y.shape)}")
        if int(y.shape[1]) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {int(y.shape[1])}")

        cfg = self.cfg
        step_x = float(cfg.step_x)
        step_u = float(cfg.step_u)
        lam_x = float(cfg.lambda_x)
        lam_u = float(cfg.lambda_u)

        # Initialize codes.
        x_old = y.new_zeros((y.shape[0], self.n_dense))
        u_old = y.new_zeros((y.shape[0], self.n_sparse))

        if bool(cfg.fista):
            x_tmp = x_old
            u_tmp = u_old
            t_old = y.new_tensor(1.0)
        else:
            x_tmp = x_old
            u_tmp = u_old

        u_pre = y.new_zeros((y.shape[0], self.n_sparse))

        for _ in range(self.n_iters):
            # Residual under current iterate.
            y_hat = self.decode(x_dense=x_tmp, u_sparse=u_tmp)
            r = y - y_hat

            x_new = x_tmp + step_x * (r @ self.A)
            if lam_x > 0.0:
                # Prox for 0.5*lam_x*||x||^2 is a shrink by 1/(1+step_x*lam_x).
                x_new = x_new / (1.0 + step_x * lam_x)

            u_pre = u_tmp + step_u * (r @ self.B)
            u_new = _soft_threshold(u_pre, step_u * lam_u, twosided=bool(cfg.twosided_u))

            if bool(cfg.fista):
                t_new = (1.0 + math.sqrt(1.0 + 4.0 * float(t_old.item()) ** 2)) / 2.0
                t_new_t = y.new_tensor(t_new)
                momentum = (t_old - 1.0) / t_new_t
                x_tmp = x_new + momentum * (x_new - x_old)
                u_tmp = u_new + momentum * (u_new - u_old)
                x_old = x_new
                u_old = u_new
                t_old = t_new_t
            else:
                x_tmp = x_new
                u_tmp = u_new

        y_hat_logits = self.decode(x_dense=x_new, u_sparse=u_new)
        return u_pre, u_new, y_hat_logits

