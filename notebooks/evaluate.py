"""
Notebook-friendly evaluation / spot-check script.

Open this file in Cursor/VSCode and run cell-by-cell via the `#%%` markers.
"""

#%%
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dense_sparse_extractor.data import denormalize_mnist
from dense_sparse_extractor.train import (
    TrainingConfig,
    _build_model,
    _dataclass_from_dict,
    _is_multilabel_targets,
    _make_loaders,
    _resolve_device,
    evaluate as eval_loop,
)


#%%
# --- User settings ---
#
# Point this to either:
# - a checkpoint file: .../epoch_0010.pt or .../latest.pt
# - a run directory: .../<checkpoint_root>/<project>/<run_name>  (we'll use latest.pt)
CKPT_PATH = "../checkpoints/dense-sparse-extractor_base_model/mlp-mnist_lowfreq/latest.pt"

# Optional override (set to "auto", "cpu", or "cuda"). If None, uses config's `device`.
DEVICE_OVERRIDE: str | None = "auto"

# How strictly to load weights. If you changed code since training, try False.
STRICT_LOAD = True

# Spot-check settings
SPLIT: str = "test"  # "train" or "test"
N_BATCHES = 3
N_SHOW = 8  # number of items to print from each batch


#%%
def _resolve_ckpt_file(path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_dir():
        latest = p / "latest.pt"
        if latest.exists():
            return latest
        # fallback: pick highest epoch_*.pt
        epoch_files = sorted(p.glob("epoch_*.pt"))
        if epoch_files:
            return epoch_files[-1]
        raise FileNotFoundError(f"No checkpoint files found in directory: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")
    return p


def _format_label(y_row: torch.Tensor) -> str:
    # For multi-hot labels, show indices with y>0.5
    if y_row.ndim == 0:
        return str(int(y_row.item()))
    if y_row.ndim == 1 and y_row.dtype.is_floating_point:
        idx = (y_row > 0.5).nonzero(as_tuple=False).flatten().tolist()
        return str(idx)
    return f"shape={tuple(y_row.shape)} dtype={y_row.dtype}"


def _format_pred(logits_row: torch.Tensor) -> str:
    if logits_row.ndim != 1:
        return f"shape={tuple(logits_row.shape)}"
    # Heuristic: if logits size is "class-like", show argmax and topk
    c = int(logits_row.numel())
    if c <= 1000:
        topk = min(5, c)
        vals, idx = torch.topk(logits_row, k=topk)
        pairs = ", ".join([f"{int(i)}:{float(v):.3f}" for i, v in zip(idx.tolist(), vals.tolist())])
        return f"argmax={int(torch.argmax(logits_row).item())} top{topk}=[{pairs}]"
    return f"numel={c}"


#%%
ckpt_file = _resolve_ckpt_file(CKPT_PATH)
print(f"Loading checkpoint: {ckpt_file}")

payload: dict[str, Any] = torch.load(ckpt_file, map_location="cpu")
epoch = int(payload.get("epoch", -1))
train_cfg_dict = dict(payload.get("training_config", {}))
model_cfg = dict(payload.get("model_config", {}))

if not train_cfg_dict:
    raise KeyError("Checkpoint payload missing `training_config`.")
if not model_cfg:
    raise KeyError("Checkpoint payload missing `model_config`.")

print(f"Checkpoint epoch: {epoch}")
print(f"Saved training_config keys: {sorted(train_cfg_dict.keys())}")
print(f"Saved model_config keys: {sorted(model_cfg.keys())}")


#%%
# Reconstruct config objects from checkpoint metadata.
train_cfg: TrainingConfig = _dataclass_from_dict(TrainingConfig, train_cfg_dict)

device_setting = DEVICE_OVERRIDE if DEVICE_OVERRIDE is not None else str(train_cfg.device)
device = _resolve_device(device_setting)
print(f"Using device: {device} (override={DEVICE_OVERRIDE!r}, config.device={train_cfg.device!r})")


#%%
# Build model from saved model_config and load weights.
model = _build_model(model_cfg).to(device)
missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=bool(STRICT_LOAD))
if not STRICT_LOAD:
    print(f"load_state_dict(strict=False) missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing keys (first 20):", missing[:20])
    if unexpected:
        print("  unexpected keys (first 20):", unexpected[:20])
model.eval()


#%%
# Recreate dataloaders using the *saved* training config.
train_loader, test_loader = _make_loaders(cfg=train_cfg, device=device)
loader = train_loader if SPLIT == "train" else test_loader
print(f"Loaded split={SPLIT!r} with {len(loader)} batches (batch_size={train_cfg.batch_size})")


#%%
# Quick aggregate metrics (same logic as training script).
train_metrics = eval_loop(model=model, loader=train_loader, device=device)
test_metrics = eval_loop(model=model, loader=test_loader, device=device)
print("train:", train_metrics)
print("test: ", test_metrics)


#%%
@torch.no_grad()
def spot_check(*, model, loader, device, n_batches: int, n_show: int) -> None:
    model.eval()
    for b_idx, (x, y) in enumerate(loader):
        if b_idx >= int(n_batches):
            break

        x = x.to(device)
        y = y.to(device)
        logits = model(x)

        print("")
        print(f"batch {b_idx} | x={tuple(x.shape)} {x.dtype} | y={tuple(y.shape)} {y.dtype} | logits={tuple(logits.shape)}")
        print(f"  x stats: min={float(x.min()):.3f} max={float(x.max()):.3f} mean={float(x.mean()):.3f} std={float(x.std()):.3f}")

        if _is_multilabel_targets(y):
            probs = torch.sigmoid(logits)
            pred = probs > 0.5
            # micro accuracy (same as train.py helper)
            micro_acc = float((pred == (y > 0.5)).float().mean().item())
            print(f"  multilabel micro-acc: {micro_acc:.4f}")
            for i in range(min(int(n_show), int(y.size(0)))):
                y_i = y[i].detach().cpu()
                pred_i = pred[i].detach().cpu()
                probs_i = probs[i].detach().cpu()
                y_idx = (y_i > 0.5).nonzero(as_tuple=False).flatten().tolist()
                p_idx = (pred_i > 0.5).nonzero(as_tuple=False).flatten().tolist()
                # show top-5 probabilities to sanity-check signal
                topk = min(5, int(probs_i.numel()))
                pv, pi = torch.topk(probs_i, k=topk)
                pairs = ", ".join([f"{int(ii)}:{float(vv):.2f}" for ii, vv in zip(pi.tolist(), pv.tolist())])
                print(f"    [{i:02d}] y={y_idx} pred={p_idx}  top{topk}={pairs}")
        else:
            pred = logits.argmax(dim=1)
            acc = float((pred == y).float().mean().item())
            print(f"  top1 acc: {acc:.4f}")
            for i in range(min(int(n_show), int(y.size(0)))):
                print(
                    f"    [{i:02d}] y={int(y[i].item())} pred={int(pred[i].item())}  logits: {_format_pred(logits[i].detach().cpu())}"
                )


spot_check(model=model, loader=loader, device=device, n_batches=N_BATCHES, n_show=N_SHOW)


#%%
# Optional: visualize a few inputs (assumes MNIST-like normalization / 1x28x28-ish tensors).
#
# If your dataset isn't MNIST-like, this cell may not be meaningful; feel free to skip.
try:
    import matplotlib.pyplot as plt

    x0, y0 = next(iter(loader))
    x0 = x0[: min(16, x0.size(0))].detach().cpu()
    y0 = y0[: min(16, y0.size(0))].detach().cpu()

    # Attempt to denormalize for display (best-effort).
    x_disp = denormalize_mnist(x0)

    n = int(x_disp.size(0))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = axes.reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= n:
            continue
        img = x_disp[i]
        if img.ndim == 3 and img.size(0) == 1:
            img = img[0]
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(_format_label(y0[i]))
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Visualization skipped ({type(e).__name__}): {e}")

