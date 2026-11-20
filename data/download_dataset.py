#!/usr/bin/env python3
"""Download and prepare the Teklia/IAM-line dataset.

Usage:
    python data/download_dataset.py

The dataset is loaded from Hugging Face Hub and cached by ``datasets`` in
``~/.cache/huggingface/hub/datasets--Teklia--IAM-line`` (≈434 MB). No raw data
is stored in this repository.

This script verifies the expected splits and column names so that downstream
training/evaluation scripts can trust the data layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.data.dataset import DATASET_NAME, load_iam_splits

EXPECTED_SPLITS = {"train": 6482, "validation": 976, "test": 2915}
EXPECTED_COLUMNS = {"image", "text"}


def main() -> None:
    print(f"Loading {DATASET_NAME} from Hugging Face Hub ...")
    hf = load_iam_splits()

    print("\n=== Splits ===")
    for split in EXPECTED_SPLITS:
        if split not in hf:
            print(f"ERROR: expected split '{split}' not found in {list(hf.keys())}")
            sys.exit(1)
        n = len(hf[split])
        ok = "OK" if n == EXPECTED_SPLITS[split] else f"MISMATCH (expected {EXPECTED_SPLITS[split]})"
        print(f"  {split:12s}: {n:6d} samples  [{ok}]")

    print("\n=== Columns ===")
    for split in EXPECTED_SPLITS:
        cols = set(hf[split].column_names)
        ok = EXPECTED_COLUMNS == cols
        print(f"  {split:12s}: {list(hf[split].column_names)}  [{'OK' if ok else 'MISMATCH'}]")

    print("\n=== Sample check (train[0]) ===")
    sample = hf["train"][0]
    image = sample["image"]
    text = sample["text"]
    print(f"  image mode : {image.mode}")
    print(f"  image size : {image.size[0]}x{image.size[1]} px")
    print(f"  text       : {text!r}")

    print("\nDataset ready. Cached by ``datasets`` in:")
    print(f"  ~/.cache/huggingface/hub/datasets--Teklia--IAM-line")
    print("\nNo raw data is stored in the repository data/ directory.")


if __name__ == "__main__":
    main()
