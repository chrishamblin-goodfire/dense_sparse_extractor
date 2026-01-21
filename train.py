"""
Compatibility shim.

Prefer running:
  python -m dense_sparse_extractor.train
"""

from dense_sparse_extractor.train import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()

