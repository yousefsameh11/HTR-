"""Evaluate the attention-decoder model on the IAM-line test split.

Writes only *new* artifacts and never rewrites the existing CRNN / TrOCR
results:

    experiments/metrics/attention_metrics.json       (this model only)
    experiments/predictions/attention_predictions.txt
    experiments/figures/attention_loss.png           (new file)
    experiments/figures/three_model_comparison.png   (new file)

``experiments/metrics/comparison.json`` is updated by adding (or replacing)
just the ``Attention HTR`` entry; the CRNN and TrOCR entries are copied through
untouched so their published numbers cannot change.

Usage (from HTR_Project/):
    python scripts/evaluate_attention.py
    python scripts/evaluate_attention.py --max-test 200 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import IAMLineDataset, load_iam_splits
from src.datasets.data.attention import AttentionCollator
from src.evaluation.evaluate import evaluate_attention, save_metrics, save_predictions
from src.evaluation.visualization import plot_loss_curves, plot_metric_group
from src.inference.attention_inference import AttentionInference

CKPT_DIR = Path("checkpoints")
MODEL_NAME = "Attention HTR"


def read_json(path: Path):
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AttentionHTR on IAM-line")
    parser.add_argument("--checkpoint", default=None,
                        help="default: checkpoints/attention/attention_best.pt")
    parser.add_argument("--vocab", default=None,
                        help="default: vocabulary embedded in the checkpoint")
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--results-dir", default="experiments",
                        help="where metrics/predictions/figures are written "
                             "(point it elsewhere for smoke runs)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics_dir = results_dir / "metrics"
    pred_dir = results_dir / "predictions"
    fig_dir = results_dir / "figures"

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested with --device cuda but torch.cuda.is_available() is False."
        )
    print(f"Device: {device}")

    checkpoint = Path(
        args.checkpoint
        or next(
            (
                path
                for path in (
                    CKPT_DIR / "attention" / "attention_best.pt",
                    CKPT_DIR / "attention" / "attention_latest.pt",
                )
                if path.is_file()
            ),
            CKPT_DIR / "attention" / "attention_best.pt",
        )
    )
    if not checkpoint.is_file():
        raise SystemExit(
            f"Attention HTR checkpoint not found: {checkpoint}\n"
            "Train it first:\n  python scripts/train_attention.py --epochs 30 --device cuda"
        )

    for directory in (metrics_dir, pred_dir, fig_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Loading {checkpoint}")
    predictor = AttentionInference(
        str(checkpoint),
        vocab_path=args.vocab,
        device=device,
        max_length=args.max_length,
    )
    model = predictor.model
    vocabulary = predictor.vocabulary
    print(f"Parameters: {model.num_parameters() / 1e6:.3f}M | vocabulary: {vocabulary.num_classes}")

    hf = load_iam_splits()
    test_split = hf["test"]
    if args.max_test:
        test_split = test_split.select(range(min(args.max_test, len(test_split))))
    print(f"Evaluating on {len(test_split)} test samples (dataset split: test)")

    dataset = IAMLineDataset(test_split, predictor.preprocessor, vocabulary=None)
    dataset.set_training(False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=AttentionCollator(vocabulary, max_length=args.max_length),
        num_workers=0,
    )

    result = evaluate_attention(
        model, loader, vocabulary, device=device, max_length=args.max_length
    )
    print(
        f"{MODEL_NAME}: CER={result['cer']:.4f}  WER={result['wer']:.4f}  "
        f"{result['avg_inference_time_per_line'] * 1000:.1f} ms/line  "
        f"n={result['num_samples']}"
    )

    save_metrics(result, str(metrics_dir / "attention_metrics.json"))
    save_predictions(result["details"], str(pred_dir / "attention_predictions.txt"))

    # ------------------------------------------------ training information
    # Read alongside the checkpoint itself so an evaluation of any run (for
    # example a smoke run in another directory) uses that run's own numbers.
    summary = read_json(checkpoint.parent / "training_summary.json") or {}
    history = read_json(checkpoint.parent / "attention_history.json") or {}

    # ---------------------------------------- comparison.json (additive only)
    comparison = read_json(metrics_dir / "comparison.json") or {}
    preserved = {k: v for k, v in comparison.items() if k != MODEL_NAME}
    preserved[MODEL_NAME] = {
        "cer": result["cer"],
        "wer": result["wer"],
        "num_samples": result["num_samples"],
        "avg_inference_time_per_line": result["avg_inference_time_per_line"],
        "total_inference_time": result["total_inference_time"],
        "val_loss": summary.get("best_val_loss"),
        "training_time_seconds": summary.get("total_time_seconds"),
    }
    comparison = preserved
    with open(metrics_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # ------------------------------------------------------------- figures
    # New filenames only: the CRNN/TrOCR figures produced by scripts/evaluate.py
    # are left exactly as they are.
    if history.get("train_loss"):
        plot_loss_curves(
            history,
            str(fig_dir / "attention_loss.png"),
            title="AttentionHTR training & validation loss",
        )
    plot_metric_group(
        {name: {"cer": v["cer"], "wer": v["wer"]} for name, v in comparison.items()},
        str(fig_dir / "three_model_comparison.png"),
        title="CER / WER - CRNN vs TrOCR vs Attention HTR",
    )

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 74)
    print("SUMMARY (same IAM-line test split for every model)")
    print("=" * 74)
    print(f"{'Model':16s} {'CER':>8s} {'WER':>8s} {'ms/line':>9s} {'train (s)':>10s}")
    for name, values in comparison.items():
        train_s = values.get("training_time_seconds")
        print(
            f"{name:16s} {values['cer'] * 100:7.2f}% {values['wer'] * 100:7.2f}% "
            f"{values['avg_inference_time_per_line'] * 1000:9.1f} "
            f"{(f'{train_s:.0f}' if train_s else 'n/a'):>10s}"
        )
    print(f"\nResults written to {results_dir.resolve()}")


if __name__ == "__main__":
    main()
