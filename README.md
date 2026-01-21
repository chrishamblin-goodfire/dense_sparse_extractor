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

