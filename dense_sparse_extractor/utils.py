from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def safe_path_component(s: str) -> str:
    """Make a readable, filesystem-friendly path component."""
    s = s.strip().replace(" ", "_")
    keep: list[str] = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    return out or "unnamed"


def pick_unique_run_name(*, checkpoint_root: str, project: str, run_name: str) -> str:
    """
    Ensure we don't overwrite existing checkpoint directories.

    If `<root>/<project>/<run_name>` already exists, return `<run_name>_v1`,
    then `_v2`, etc. Uniqueness is checked on the filesystem-safe folder names.
    """
    root = Path(checkpoint_root)
    project_dir = root / safe_path_component(project)

    base = run_name.strip() or "unnamed"
    max_tries = 10_000
    for i in range(max_tries):
        candidate = base if i == 0 else f"{base}_v{i}"
        candidate_dir = project_dir / safe_path_component(candidate)
        if not candidate_dir.exists():
            return candidate
    raise RuntimeError(f"Could not find a unique run name after {max_tries} attempts.")


def checkpoint_dir(*, checkpoint_root: str, project: str, name: str) -> Path:
    """Compute checkpoint directory `<root>/<project>/<name>`."""
    return Path(checkpoint_root) / safe_path_component(project) / safe_path_component(name)


def save_checkpoint(
    *,
    ckpt_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_cfg_dict: dict[str, Any],
    model_cfg: dict[str, Any],
    device: torch.device,
) -> Path:
    """
    Save full training state (model + optimizer) so training can be resumed later.

    Writes `epoch_XXXX.pt` and also updates `latest.pt`.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"

    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_config": train_cfg_dict,
        "model_config": model_cfg,
        "torch_rng_state": torch.get_rng_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    torch.save(payload, ckpt_path)
    torch.save(payload, ckpt_dir / "latest.pt")
    return ckpt_path

