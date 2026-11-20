"""Inspect the Teklia/IAM-line Hugging Face dataset.

Prints:
  - Available splits
  - Column names per split
  - Feature types (image vs text columns)
  - Number of samples per split
  - Several image-text examples (saved to disk + metadata printed)
  - Verification that every sample has a valid transcription
  - Presence of writer-independent split information (writer_id column, etc.)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Any, Dict, List

from datasets import load_dataset, DatasetDict
from PIL import Image

# Allow running both as a script and as a module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "inspection_examples")


def load_iam_dataset() -> DatasetDict:
    print("Loading Teklia/IAM-line dataset from Hugging Face...")
    dataset = load_dataset("Teklia/IAM-line")
    print(f"Dataset type: {type(dataset)}")
    return dataset


def print_splits(dataset: DatasetDict) -> None:
    print("\n" + "=" * 70)
    print("AVAILABLE SPLITS")
    print("=" * 70)
    if isinstance(dataset, DatasetDict):
        for split_name in dataset.keys():
            split = dataset[split_name]
            print(f"  - {split_name}: {len(split)} samples, {split.num_columns} columns")
    else:
        print(f"  Single dataset: {len(dataset)} samples")


def print_columns(dataset: DatasetDict) -> None:
    print("\n" + "=" * 70)
    print("COLUMN NAMES & FEATURE TYPES")
    print("=" * 70)
    for split_name in dataset.keys():
        split = dataset[split_name]
        print(f"\nSplit: {split_name}")
        features = split.features
        for col_name in split.column_names:
            feat = features[col_name]
            print(f"  - {col_name}: {feat}")


def find_text_and_image_columns(dataset: DatasetDict) -> tuple[str, str]:
    """Heuristically detect image and text columns."""
    split = dataset[list(dataset.keys())[0]]
    image_col = None
    text_col = None
    for col_name in split.column_names:
        feat = split.features[col_name]
        feat_str = str(feat).lower()
        if "image" in feat_str and image_col is None:
            image_col = col_name
        if col_name.lower() in ("text", "transcription", "label", "sentence") or "text" in feat_str:
            if text_col is None:
                text_col = col_name
    return image_col, text_col


def verify_transcriptions(dataset: DatasetDict, text_column: str) -> Dict[str, Any]:
    """Verify every sample has a valid (non-empty) transcription."""
    report: Dict[str, Any] = {}
    for split_name in dataset.keys():
        split = dataset[split_name]
        total = len(split)
        missing = 0
        empty = 0
        lengths: List[int] = []
        for i in range(total):
            try:
                value = split[i][text_column]
            except Exception:  # pragma: no cover - defensive
                missing += 1
                continue
            if value is None:
                missing += 1
            elif isinstance(value, str) and value.strip() == "":
                empty += 1
            else:
                lengths.append(len(str(value)))
        report[split_name] = {
            "total": total,
            "missing": missing,
            "empty": empty,
            "valid": total - missing - empty,
            "min_len": min(lengths) if lengths else 0,
            "max_len": max(lengths) if lengths else 0,
            "avg_len": sum(lengths) / len(lengths) if lengths else 0,
        }
    return report


def check_writer_splits(dataset: DatasetDict) -> None:
    """Check for writer-independent split information."""
    print("\n" + "=" * 70)
    print("WRITER-INDEPENDENT SPLIT CHECK")
    print("=" * 70)
    for split_name in dataset.keys():
        split = dataset[split_name]
        writer_cols = [c for c in split.column_names if "writer" in c.lower() or "id" in c.lower()]
        if writer_cols:
            print(f"Split '{split_name}' candidate writer/id columns: {writer_cols}")
            for col in writer_cols:
                try:
                    values = split[col]
                    unique = len(set(values)) if hasattr(values, "__iter__") else "n/a"
                    print(f"  - {col}: {unique} unique values (sample: {values[:5]})")
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"  - {col}: error reading ({exc})")
        else:
            print(f"Split '{split_name}': no obvious writer/id columns found.")


def save_examples(dataset: DatasetDict, image_column: str, text_column: str, n: int = 5) -> None:
    """Save a few image-text examples to disk for visual inspection."""
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    print("\n" + "=" * 70)
    print(f"EXAMPLES (first {n} train samples, saved to {EXAMPLES_DIR})")
    print("=" * 70)
    split = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]
    metadata_path = os.path.join(EXAMPLES_DIR, "examples.txt")
    with open(metadata_path, "w", encoding="utf-8") as meta:
        meta.write("index\ttext\n")
        for i in range(min(n, len(split))):
            sample = split[i]
            image = sample[image_column]
            text = sample[text_column]
            if not isinstance(image, Image.Image):
                try:
                    image = Image.fromarray(image)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"  Skipping example {i}: cannot convert image ({exc})")
                    continue
            img_path = os.path.join(EXAMPLES_DIR, f"example_{i:02d}.png")
            image.save(img_path)
            line = f"{i}\t{text}"
            meta.write(line + "\n")
            print(f"  [{i}] image size={image.size}, mode={image.mode} | text='{text}'")
    print(f"  Metadata written to {metadata_path}")


def print_summary(dataset: DatasetDict, image_column: str, text_column: str,
                  transcription_report: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Detected image column : {image_column}")
    print(f"Detected text column  : {text_column}")
    for split_name, info in transcription_report.items():
        print(f"\nSplit '{split_name}':")
        print(f"  total samples : {info['total']}")
        print(f"  valid text    : {info['valid']}")
        print(f"  missing       : {info['missing']}")
        print(f"  empty         : {info['empty']}")
        if info["valid"] > 0:
            print(f"  text length   : min={info['min_len']}, max={info['max_len']}, avg={info['avg_len']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Teklia/IAM-line dataset")
    parser.add_argument("--examples", type=int, default=5, help="Number of example images to save")
    args = parser.parse_args()

    dataset = load_iam_dataset()
    print_splits(dataset)
    print_columns(dataset)

    image_column, text_column = find_text_and_image_columns(dataset)
    print(f"\nHeuristic detection -> image: {image_column}, text: {text_column}")

    transcription_report = verify_transcriptions(dataset, text_column)
    check_writer_splits(dataset)
    save_examples(dataset, image_column, text_column, n=args.examples)
    print_summary(dataset, image_column, text_column, transcription_report)


if __name__ == "__main__":
    main()