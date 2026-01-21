from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import json
import shutil
import sys
import time
import torch
import torch.nn as nn
from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import NoiseTagConfig, make_combined_loaders, make_mnist_loaders, make_noise_tag_loaders
from .models.convnet import ConvNetConfig, MNISTConvNet
from .models.mlp import MLPConfig, MNISTMLP
from .models.vit import ViTConfig, MNISTViT
from .utils import checkpoint_dir, pick_unique_run_name, save_checkpoint, write_yaml


ModelType = Literal["mlp", "convnet", "vit"]
DatasetType = Literal["mnist", "noise_tags", "combined"]


@dataclass(frozen=True)
class TrainingConfig:
    # Used for checkpoint folder naming (and W&B project/run names if enabled)
    project_name: str | None = None
    run_name: str | None = None
    wandb_enabled: bool = False

    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True

    # Data selection
    dataset: DatasetType = "mnist"  # "mnist" | "noise_tags" | "combined"
    # Extra dataset-specific config blobs (parsed from YAML)
    noise_tags: dict[str, Any] = field(default_factory=dict)
    combined: dict[str, Any] = field(default_factory=dict)

    optimizer: str = "adamw"  # "adamw" | "adam" | "sgd"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    momentum: float = 0.9

    n_epochs: int = 5
    log_every_steps: int = 100

    # Checkpointing
    # Saves full training state (model + optimizer) every N epochs.
    # Set to 0 to disable.
    checkpoint_every_epochs: int = 0
    checkpoint_root: str = "./checkpoints"


@dataclass(frozen=True)
class ModelArgs:
    """
    A flat superset of model hyperparameters across MLP/ConvNet/ViT.

    jsonargparse needs a single, non-conflicting set of CLI flags; we then
    filter these down into the chosen model's dataclass.
    """

    model_type: ModelType = "mlp"

    # Common
    num_classes: int = 10
    dropout: float = 0.1
    activation: str = "gelu"

    # MLP
    in_dim: int = 28 * 28
    hidden_sizes: tuple[int, ...] = (256, 256)

    # ConvNet (and ViT input)
    in_channels: int = 1

    # ConvNet
    channels: tuple[int, ...] = (32, 64)
    kernel_size: int = 3
    use_batchnorm: bool = True
    head_hidden_dim: int = 128

    # ViT
    image_size: int = 28
    patch_size: int = 4
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    attn_dropout: float = 0.1
    layer_norm_eps: float = 1e-5


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


def _merged_noise_cfg(noise_cfg_dict: dict[str, Any]) -> NoiseTagConfig:
    """
    Create a NoiseTagConfig from a (possibly partial) YAML mapping.
    Unknown keys are ignored to keep configs forward-compatible.
    """
    allowed = {f.name for f in fields(NoiseTagConfig)}
    filtered = {k: v for k, v in dict(noise_cfg_dict).items() if k in allowed}
    return NoiseTagConfig(**filtered)


def _make_loaders(*, cfg: TrainingConfig, device: torch.device) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(cfg.data_dir)

    if cfg.dataset == "mnist":
        return make_mnist_loaders(
            data_dir=data_dir,
            batch_size=int(cfg.batch_size),
            shuffle_train=True,
            num_workers=int(cfg.num_workers),
            pin_memory=bool(cfg.pin_memory),
            normalize=True,
            label_format="int",
            download=True,
            device=device,
        )

    if cfg.dataset == "noise_tags":
        noise_cfg = _merged_noise_cfg(cfg.noise_tags)
        return make_noise_tag_loaders(
            cfg=noise_cfg,
            batch_size=int(cfg.batch_size),
            shuffle_train=True,
            num_workers=int(cfg.num_workers),
            pin_memory=bool(cfg.pin_memory),
            device=device,
        )

    if cfg.dataset == "combined":
        combined_cfg = dict(cfg.combined)
        combined_seed = int(combined_cfg.get("seed", 0))
        combined_length = combined_cfg.get("length", None)
        if combined_length is not None:
            combined_length = int(combined_length)
        clip = combined_cfg.get("clip", None)
        clip_tuple: tuple[float, float] | None = None
        if clip is not None:
            if not (isinstance(clip, (list, tuple)) and len(clip) == 2):
                raise ValueError("combined.clip must be null or a 2-element list/tuple like: [lo, hi]")
            clip_tuple = (float(clip[0]), float(clip[1]))

        # Allow combined.noise_tags to override cfg.noise_tags
        noise_cfg_dict = dict(cfg.noise_tags)
        if isinstance(combined_cfg.get("noise_tags", None), dict):
            noise_cfg_dict.update(dict(combined_cfg["noise_tags"]))
        # We enforce one-hot for combined downstream, but keep config parsing consistent.
        noise_cfg = _merged_noise_cfg(noise_cfg_dict)

        return make_combined_loaders(
            data_dir=data_dir,
            noise_cfg=noise_cfg,
            batch_size=int(cfg.batch_size),
            shuffle_train=True,
            num_workers=int(cfg.num_workers),
            pin_memory=bool(cfg.pin_memory),
            mnist_normalize=True,
            download=True,
            seed=combined_seed,
            length=combined_length,
            clip=clip_tuple,
            device=device,
        )

    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def _is_multilabel_targets(y: torch.Tensor) -> bool:
    # multi-label / (multi-)one-hot targets should be floating vectors: (B, C)
    return torch.is_tensor(y) and y.dtype.is_floating_point and y.ndim == 2


def _accuracy_single_label(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


def _accuracy_multilabel(logits: torch.Tensor, y: torch.Tensor) -> float:
    # Elementwise label accuracy (micro-average across batch and labels).
    y_true = y > 0.5
    y_pred = torch.sigmoid(logits) > 0.5
    return float((y_pred == y_true).float().mean().item())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        if _is_multilabel_targets(y):
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, y)
            acc = _accuracy_multilabel(logits, y)
        else:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, y)
            acc = _accuracy_single_label(logits, y)

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        total_acc += float(acc) * bs
        total += bs

    return {
        "loss": total_loss / max(1, total),
        "acc": total_acc / max(1, total),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every_steps: int,
    *,
    wandb_run: Any | None = None,
    epoch: int | None = None,
    global_step: int = 0,
) -> tuple[dict[str, float], dict[str, float], int]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    epoch_t0 = time.perf_counter()
    batch_times_sec: list[float] = []
    interval_sum_sec = 0.0
    interval_count = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for step, (x, y) in enumerate(pbar, start=1):
        batch_t0 = time.perf_counter()

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        if _is_multilabel_targets(y):
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, y)
            acc = _accuracy_multilabel(logits, y)
        else:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, y)
            acc = _accuracy_single_label(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = int(y.size(0))
        total_loss += float(loss.item()) * bs
        total_acc += float(acc) * bs
        total += bs

        batch_t1 = time.perf_counter()
        batch_time_sec = float(batch_t1 - batch_t0)
        batch_times_sec.append(batch_time_sec)
        interval_sum_sec += batch_time_sec
        interval_count += 1

        global_step += 1
        if wandb_run is not None and log_every_steps and step % int(log_every_steps) == 0:
            # Respect the same cadence as console logging.
            # We log the mean batch time over the last interval for stability.
            wandb_run.log(
                {
                    "epoch": int(epoch) if epoch is not None else 0,
                    "global_step": int(global_step),
                    "time/batch_sec": float(interval_sum_sec / max(1, interval_count)),
                },
                step=int(global_step),
            )
            interval_sum_sec = 0.0
            interval_count = 0

        if log_every_steps and step % int(log_every_steps) == 0:
            pbar.set_postfix(loss=total_loss / max(1, total), acc=total_acc / max(1, total))

    epoch_t1 = time.perf_counter()
    epoch_time_sec = float(epoch_t1 - epoch_t0)
    avg_batch_time_sec = float(sum(batch_times_sec) / max(1, len(batch_times_sec)))

    metrics = {
        "loss": total_loss / max(1, total),
        "acc": total_acc / max(1, total),
    }
    timing = {
        "time/epoch_sec": epoch_time_sec,
        "time/avg_batch_sec": avg_batch_time_sec,
    }

    return metrics, timing, global_step


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


def _save_config_snapshot(
    *,
    ckpt_dir: Path,
    train_cfg_dict: dict[str, Any],
    model_cfg: dict[str, Any],
    full_cfg_dict: dict[str, Any],
    config_files: list[str],
    argv: list[str],
) -> Path:
    """
    Save an exact, reproducible snapshot of the run configuration.

    Includes:
    - resolved training/model YAML (after all config merging + CLI overrides)
    - full resolved YAML (union)
    - the raw source YAML files (defaults + any --config files)
    - the exact argv used to launch the run
    """
    snap_dir = ckpt_dir / "config_snapshot"
    sources_dir = snap_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    (snap_dir / "argv.txt").write_text(" ".join([sys.executable, *argv]) + "\n", encoding="utf-8")
    (snap_dir / "config_files.txt").write_text("\n".join(config_files) + "\n", encoding="utf-8")

    write_yaml(snap_dir / "training.resolved.yaml", train_cfg_dict)
    write_yaml(snap_dir / "model.resolved.yaml", model_cfg)
    write_yaml(snap_dir / "full.resolved.yaml", full_cfg_dict)

    manifest: list[dict[str, str]] = []
    for i, p in enumerate(config_files):
        src = Path(p)
        if not src.exists() or not src.is_file():
            manifest.append({"src": str(src), "dst": ""})
            continue
        dst = sources_dir / f"{i:02d}_{src.name}"
        shutil.copy2(src, dst)
        manifest.append({"src": str(src), "dst": str(dst)})
    (snap_dir / "sources_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return snap_dir


@torch.no_grad()
def _weight_l2_norm(model: nn.Module) -> float:
    total_sq = 0.0
    for p in model.parameters():
        v = p.detach()
        total_sq += float(v.float().pow(2).sum().item())
    return float(total_sq**0.5)


def main() -> None:
    # Defaults point at repo-root configs when running `python -m dense_sparse_extractor.train`.
    repo_root = Path(__file__).resolve().parents[1]
    default_model = repo_root / "configs" / "model" / "baseline_mlp.yaml"
    default_train = repo_root / "configs" / "training" / "baseline.yaml"

    parser = ArgumentParser(
        default_config_files=[str(default_model), str(default_train)],
    )
    # Optional additional config files. Can be provided multiple times; later ones override earlier ones.
    # Note: jsonargparse only allows a single config-file argument per parser.
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to a YAML config file (model or training). Can be given multiple times.",
    )

    # Flat arguments matching the existing YAML structure.
    parser.add_class_arguments(TrainingConfig, nested_key=None, as_group=True)
    parser.add_class_arguments(ModelArgs, nested_key=None, as_group=True)

    cfg_ns = parser.parse_args()
    cfg_dict = namespace_to_dict(cfg_ns)

    extra_cfg_files = getattr(cfg_ns, "config", None)
    if extra_cfg_files is None:
        extra_cfg_list: list[str] = []
    elif isinstance(extra_cfg_files, (list, tuple)):
        extra_cfg_list = [str(x) for x in extra_cfg_files]
    else:
        extra_cfg_list = [str(extra_cfg_files)]
    config_files = [str(default_model), str(default_train), *extra_cfg_list]
    argv = sys.argv[1:]

    train_keys = {f.name for f in fields(TrainingConfig)}
    model_keys = (
        {f.name for f in fields(MLPConfig)}
        | {f.name for f in fields(ConvNetConfig)}
        | {f.name for f in fields(ViTConfig)}
        | {"model_type"}
    )

    train_cfg_dict = {k: v for k, v in cfg_dict.items() if k in train_keys}
    model_cfg = {k: v for k, v in cfg_dict.items() if k in model_keys}
    train_cfg = _dataclass_from_dict(TrainingConfig, train_cfg_dict)

    torch.manual_seed(int(train_cfg.seed))

    device = _resolve_device(train_cfg.device)
    print(f"Using device: {device}")

    train_loader, test_loader = _make_loaders(cfg=train_cfg, device=device)

    model = _build_model(model_cfg).to(device)
    optimizer = _make_optimizer(model, train_cfg)

    # We always require these, since checkpoint folders are organized by them.
    if not train_cfg.project_name:
        raise ValueError("Missing `project_name` in the train config YAML (required for checkpoint folder naming).")
    if not train_cfg.run_name:
        raise ValueError("Missing `run_name` in the train config YAML (required for checkpoint folder naming).")
    project_name = str(train_cfg.project_name)
    base_run_name = str(train_cfg.run_name)

    # Ensure uniqueness of checkpoint folder (and thus uniqueness of run name).
    effective_run_name = pick_unique_run_name(
        checkpoint_root=train_cfg.checkpoint_root,
        project=project_name,
        run_name=base_run_name,
    )

    wandb_run = None
    if bool(train_cfg.wandb_enabled):
        try:
            import wandb  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "W&B logging is enabled in the config, but wandb is not installed. "
                "Install it with `pip install wandb` (or add it to requirements)."
            ) from e

        wandb_run = wandb.init(
            project=project_name,
            name=effective_run_name,
            config={
                "training": train_cfg_dict,
                "model": model_cfg,
                "cli": {"argv": argv, "config_files": config_files},
            },
        )
        # Make W&B plots less confusing:
        # - epoch-level metrics plot against `epoch`
        # - batch timing plots against `global_step`
        wandb.define_metric("epoch")
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        wandb.define_metric("model/*", step_metric="epoch")
        wandb.define_metric("time/epoch_sec", step_metric="epoch")
        wandb.define_metric("time/avg_batch_sec", step_metric="epoch")
        wandb.define_metric("time/batch_sec", step_metric="global_step")
        wandb_run.log({"device": str(device)})

    ckpt_dir = checkpoint_dir(
        checkpoint_root=train_cfg.checkpoint_root,
        project=project_name,
        name=effective_run_name,
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    snap_dir = _save_config_snapshot(
        ckpt_dir=ckpt_dir,
        train_cfg_dict=train_cfg_dict,
        model_cfg=model_cfg,
        full_cfg_dict=cfg_dict,
        config_files=config_files,
        argv=argv,
    )

    if wandb_run is not None:
        import wandb  # type: ignore

        artifact = wandb.Artifact(name=f"{effective_run_name}-config", type="config")
        artifact.add_dir(str(snap_dir))
        wandb_run.log_artifact(artifact)

    global_step = 0
    for epoch in range(1, int(train_cfg.n_epochs) + 1):
        train_metrics, train_timing, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            log_every_steps=int(train_cfg.log_every_steps),
            wandb_run=wandb_run,
            epoch=int(epoch),
            global_step=int(global_step),
        )
        test_metrics = evaluate(model=model, loader=test_loader, device=device)
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"test loss {test_metrics['loss']:.4f} acc {test_metrics['acc']:.4f}"
        )

        if wandb_run is not None:
            weight_norm = _weight_l2_norm(model)
            wandb_run.log(
                {
                    "epoch": epoch,
                    "global_step": int(global_step),
                    "train/loss": float(train_metrics["loss"]),
                    "train/acc": float(train_metrics["acc"]),
                    "test/loss": float(test_metrics["loss"]),
                    "test/acc": float(test_metrics["acc"]),
                    "model/weight_norm_l2": float(weight_norm),
                    **{k: float(v) for k, v in train_timing.items()},
                },
                step=int(global_step),
            )

        if int(train_cfg.checkpoint_every_epochs) > 0 and epoch % int(train_cfg.checkpoint_every_epochs) == 0:
            ckpt_path = save_checkpoint(
                ckpt_dir=ckpt_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_cfg_dict=train_cfg_dict,
                model_cfg=model_cfg,
                device=device,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

