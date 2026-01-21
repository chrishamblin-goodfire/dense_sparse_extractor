## Overview
For a full overview of this project, [see this notion page](https://www.notion.so/goodfire/Dense-Sparse-Feature-Extraction-2eef566bfbc180b48acffbba7f904c28?showMoveTo=true&saveParent=true). It combines in a single document, ideation for this project, the exact code that was run, the results and their interpretation.

## Setup

Create a virtual environment with `uv` and install dependencies:

```bash
cd /mnt/polished-lake/home/chamblin/projects/dense_sparse_extractor
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
python -m ipykernel install --user --name=dense_sparse_extractor --display-name="Dense Sparse Extractor"
```

## Training

Run training with the default configs:

```bash
python -m dense_sparse_extractor.train
```

Or specify model/training config files:

```bash
python -m dense_sparse_extractor.train \
  --model_config configs/model/baseline_mlp.yaml \
  --train_config configs/training/baseline.yaml
```

## Weights & Biases (wandb)

W&B logging is **optional** and configured via the training YAML under `wandb:`.

- **Minimal config** (enables logging):
  - Set `wandb.enabled: true`
  - Set `wandb.project: <your-project>`

Example `configs/training/my_run.yaml`:

```yaml
seed: 42
device: auto
data_dir: ./data
batch_size: 128
num_workers: 2
pin_memory: true

optimizer: adamw
lr: 0.001
weight_decay: 0.01
momentum: 0.9

n_epochs: 5
log_every_steps: 100

wandb:
  enabled: true
  project: dense-sparse-extractor
  name: mlp-baseline-run-1
  # entity: your_team_or_user
  # tags: ["baseline", "mnist"]
  # notes: "first run"
  # mode: "offline"   # or "online" / "disabled"
```

Then run:

```bash
python -m dense_sparse_extractor.train --train_config configs/training/my_run.yaml
```

