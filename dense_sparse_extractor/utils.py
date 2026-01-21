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


def yaml_scalar(x: Any) -> str:
    # YAML 1.2 is a superset of JSON, so JSON scalars/strings are valid YAML.
    import json

    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return repr(x)
    if isinstance(x, str):
        return json.dumps(x)
    return json.dumps(str(x))


def yaml_key(k: Any) -> str:
    import json

    s = str(k)
    if s and all(ch.isalnum() or ch in ("_", "-", ".") for ch in s):
        return s
    return json.dumps(s)


def yaml_dump_lines(obj: Any, *, indent: int = 0) -> list[str]:
    sp = "  " * int(indent)

    if isinstance(obj, dict):
        if not obj:
            return [sp + "{}"]
        out: list[str] = []
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            v = obj[k]
            key = yaml_key(k)
            if isinstance(v, dict):
                if not v:
                    out.append(f"{sp}{key}: {{}}")
                else:
                    out.append(f"{sp}{key}:")
                    out.extend(yaml_dump_lines(v, indent=indent + 1))
            elif isinstance(v, (list, tuple)):
                if len(v) == 0:
                    out.append(f"{sp}{key}: []")
                else:
                    out.append(f"{sp}{key}:")
                    out.extend(yaml_dump_lines(list(v), indent=indent + 1))
            else:
                out.append(f"{sp}{key}: {yaml_scalar(v)}")
        return out

    if isinstance(obj, (list, tuple)):
        seq = list(obj)
        if not seq:
            return [sp + "[]"]
        out: list[str] = []
        for item in seq:
            if isinstance(item, dict):
                if not item:
                    out.append(f"{sp}- {{}}")
                else:
                    out.append(f"{sp}-")
                    out.extend(yaml_dump_lines(item, indent=indent + 1))
            elif isinstance(item, (list, tuple)):
                if len(item) == 0:
                    out.append(f"{sp}- []")
                else:
                    out.append(f"{sp}-")
                    out.extend(yaml_dump_lines(list(item), indent=indent + 1))
            else:
                out.append(f"{sp}- {yaml_scalar(item)}")
        return out

    return [sp + yaml_scalar(obj)]


def write_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(yaml_dump_lines(obj)) + "\n"
    path.write_text(text, encoding="utf-8")

