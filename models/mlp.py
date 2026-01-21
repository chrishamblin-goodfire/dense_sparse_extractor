from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn


ActivationName = Literal["relu", "gelu", "tanh"]


def _activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


@dataclass(frozen=True)
class MLPConfig:
    """Hyperparameters for an MNIST MLP classifier."""

    in_dim: int = 28 * 28
    hidden_sizes: Sequence[int] = (256, 256)
    num_classes: int = 10
    dropout: float = 0.1
    activation: ActivationName = "gelu"


class MNISTMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        d_in = cfg.in_dim
        for d_out in cfg.hidden_sizes:
            layers.append(nn.Linear(d_in, int(d_out)))
            layers.append(_activation(cfg.activation))
            if cfg.dropout and cfg.dropout > 0:
                layers.append(nn.Dropout(p=float(cfg.dropout)))
            d_in = int(d_out)

        layers.append(nn.Linear(d_in, cfg.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        x = x.view(x.shape[0], -1)
        return self.net(x)

