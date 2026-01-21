from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ConvNetConfig:
    """Hyperparameters for a small MNIST ConvNet classifier."""

    in_channels: int = 1
    num_classes: int = 10

    channels: Sequence[int] = (32, 64)
    kernel_size: int = 3
    use_batchnorm: bool = True
    dropout: float = 0.1

    head_hidden_dim: int = 128


class MNISTConvNet(nn.Module):
    def __init__(self, cfg: ConvNetConfig):
        super().__init__()
        self.cfg = cfg

        conv_blocks: list[nn.Module] = []
        c_in = cfg.in_channels
        for c_out in cfg.channels:
            c_out = int(c_out)
            conv_blocks.append(
                nn.Conv2d(
                    c_in,
                    c_out,
                    kernel_size=int(cfg.kernel_size),
                    stride=1,
                    padding=int(cfg.kernel_size) // 2,
                    bias=not cfg.use_batchnorm,
                )
            )
            if cfg.use_batchnorm:
                conv_blocks.append(nn.BatchNorm2d(c_out))
            conv_blocks.append(nn.ReLU(inplace=True))
            conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if cfg.dropout and cfg.dropout > 0:
                conv_blocks.append(nn.Dropout2d(p=float(cfg.dropout)))
            c_in = c_out

        self.features = nn.Sequential(*conv_blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in, int(cfg.head_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)) if cfg.dropout and cfg.dropout > 0 else nn.Identity(),
            nn.Linear(int(cfg.head_hidden_dim), cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)

