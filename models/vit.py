from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ViTConfig:
    """Hyperparameters for a small ViT classifier for MNIST."""

    image_size: int = 28
    patch_size: int = 7
    in_channels: int = 1
    num_classes: int = 10

    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0

    dropout: float = 0.1
    attn_dropout: float = 0.1

    layer_norm_eps: float = 1e-5
    activation: Literal["gelu", "relu"] = "gelu"


class MNISTViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.image_size % cfg.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size for this ViT.")

        grid = cfg.image_size // cfg.patch_size
        num_patches = grid * grid

        self.patch_embed = nn.Conv2d(
            cfg.in_channels,
            cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=True,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, cfg.embed_dim))
        self.pos_drop = nn.Dropout(p=float(cfg.dropout))

        act = "gelu" if cfg.activation == "gelu" else "relu"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=int(cfg.embed_dim * cfg.mlp_ratio),
            dropout=float(cfg.dropout),
            activation=act,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.depth)

        self.norm = nn.LayerNorm(cfg.embed_dim, eps=float(cfg.layer_norm_eps))
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        self._init_params()

    def _init_params(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        x = self.patch_embed(x)  # (B, E, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)

        cls = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, E)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, E)
        x = self.pos_drop(x + self.pos_embed)

        x = self.encoder(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

