from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import make_mnist_loaders
from .models.convnet import ConvNetConfig, MNISTConvNet
from .models.mlp import MLPConfig, MNISTMLP
from .models.vit import ViTConfig, MNISTViT


ModelType = Literal["mlp", "convnet", "vit"]


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True

    optimizer: str = "adamw"  # "adamw" | "adam" | "sgd"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    momentum: float = 0.9

    n_epochs: int = 5
    log_every_steps: int = 100


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping/dict.")
    return dict(data)


def _dataclass_from_dict(cls: type[Any], d: dict[str, Any]) -> Any:
    allowed = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in d.items() if k in allowed}
    return cls(**filtered)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device)
    raise ValueError(f"Unknown device setting: {device}")


def _build_model(model_cfg: dict[str, Any]) -> nn.Module:
    model_type = model_cfg.get("model_type", None)
    if model_type not in ("mlp", "convnet", "vit"):
        raise ValueError("model_type must be one of: mlp, convnet, vit")

    cfg_wo_type = {k: v for k, v in model_cfg.items() if k != "model_type"}

    if model_type == "mlp":
        cfg = _dataclass_from_dict(MLPConfig, cfg_wo_type)
        return MNISTMLP(cfg)
    if model_type == "convnet":
        cfg = _dataclass_from_dict(ConvNetConfig, cfg_wo_type)
        return MNISTConvNet(cfg)
    if model_type == "vit":
        cfg = _dataclass_from_dict(ViTConfig, cfg_wo_type)
        return MNISTViT(cfg)

    raise AssertionError("Unreachable")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.size(0))

    return {
        "loss": total_loss / max(1, total),
        "acc": total_correct / max(1, total),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every_steps: int,
) -> dict[str, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for step, (x, y) in enumerate(pbar, start=1):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += bs

        if log_every_steps and step % int(log_every_steps) == 0:
            pbar.set_postfix(loss=total_loss / max(1, total), acc=total_correct / max(1, total))

    return {
        "loss": total_loss / max(1, total),
        "acc": total_correct / max(1, total),
    }


def _make_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    opt = cfg.optimizer.lower()
    params = model.parameters()

    if opt == "adamw":
        return torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    if opt == "adam":
        return torch.optim.Adam(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    if opt == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(cfg.lr),
            momentum=float(cfg.momentum),
            weight_decay=float(cfg.weight_decay),
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def main() -> None:
    # Defaults point at repo-root configs when running `python -m dense_sparse_extractor.train`.
    repo_root = Path(__file__).resolve().parents[1]
    default_model = repo_root / "configs" / "model" / "baseline_mlp.yaml"
    default_train = repo_root / "configs" / "training" / "baseline.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default=str(default_model))
    parser.add_argument("--train_config", type=str, default=str(default_train))
    args = parser.parse_args()

    model_cfg = _load_yaml(Path(args.model_config))
    train_cfg_dict = _load_yaml(Path(args.train_config))
    train_cfg = _dataclass_from_dict(TrainingConfig, train_cfg_dict)

    torch.manual_seed(int(train_cfg.seed))

    device = _resolve_device(train_cfg.device)
    print(f"Using device: {device}")

    train_loader, test_loader = make_mnist_loaders(
        data_dir=Path(train_cfg.data_dir),
        batch_size=int(train_cfg.batch_size),
        shuffle_train=True,
        num_workers=int(train_cfg.num_workers),
        pin_memory=bool(train_cfg.pin_memory),
        normalize=True,
        label_format="int",
        download=True,
        device=device,
    )

    model = _build_model(model_cfg).to(device)
    optimizer = _make_optimizer(model, train_cfg)

    for epoch in range(1, int(train_cfg.n_epochs) + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            log_every_steps=int(train_cfg.log_every_steps),
        )
        test_metrics = evaluate(model=model, loader=test_loader, device=device)
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"test loss {test_metrics['loss']:.4f} acc {test_metrics['acc']:.4f}"
        )


if __name__ == "__main__":
    main()

