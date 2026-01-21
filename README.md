## Setup

Create a virtual environment with `uv` and install dependencies:

```bash
cd /mnt/polished-lake/home/chamblin/projects/dense_sparse_extractor
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name=dense_sparse_extractor --display-name="Dense Sparse Extractor"
```

