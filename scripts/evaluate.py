"""Evaluate CRNN and TrOCR on the same IAM test samples and save real results.

Usage (VS Code terminal, from HTR_Project/):
    python scripts/evaluate.py --max-test 400
    python scripts/evaluate.py --crnn-checkpoint checkpoints/crnn/crnn_best.pt \
        --trocr-dir checkpoints/trocr/final --max-test 400

Metrics (CER, WER, inference time) come from the checkpoints on disk; nothing is
hard-coded. A model whose checkpoint is missing is skipped and reported as such.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.collate import crnn_collate_fn
from src.data.dataset import IAMLineDataset, load_iam_splits
from src.evaluation.evaluate import (
    evaluate_crnn,
    evaluate_trocr,
    save_metrics,
    save_predictions,
)
from src.evaluation.visualization import (
    make_all_figures,
    plot_predictions_with_images,
    plot_time_comparison,
)
from src.inference.predict import CRNNInference
from src.training.losses import make_ctc_loss

RESULTS_DIR = Path("experiments")
METRICS_DIR = RESULTS_DIR / "metrics"
PRED_DIR = RESULTS_DIR / "predictions"
FIG_DIR = RESULTS_DIR / "figures"
CKPT_DIR = Path("checkpoints")


def _read_json(path: Path):
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _first_existing(*paths: Path):
    for path in paths:
        if path.is_file():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CRNN and TrOCR on IAM-line")
    parser.add_argument("--crnn-checkpoint", default=None)
    parser.add_argument("--crnn-vocab", default=None)
    parser.add_argument("--trocr-dir", default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--trocr-batch-size", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=1, help="CRNN CTC beam width (1 = greedy)")
    parser.add_argument("--trocr-beams", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    crnn_ckpt = args.crnn_checkpoint or _first_existing(
        CKPT_DIR / "crnn/crnn_best.pt", CKPT_DIR / "crnn/crnn_latest.pt"
    )
    crnn_vocab = args.crnn_vocab or _first_existing(CKPT_DIR / "crnn/vocab.json")
    trocr_dir = args.trocr_dir or (
        str(CKPT_DIR / "trocr/final") if (CKPT_DIR / "trocr/final/config.json").is_file() else None
    )

    for directory in (METRICS_DIR, PRED_DIR, FIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    hf = load_iam_splits()
    test_split = hf["test"]
    if args.max_test:
        test_split = test_split.select(range(min(args.max_test, len(test_split))))
    print(f"Evaluating on {len(test_split)} test samples (dataset split: test)")

    metrics: dict[str, dict] = {}

    # ------------------------------------------------------------------ CRNN
    crnn_result = None
    if crnn_ckpt is None:
        print("CRNN: no checkpoint found -> SKIPPED (results marked unavailable).")
    else:
        print(f"\n=== CRNN: {crnn_ckpt} ===")
        predictor = CRNNInference(
            str(crnn_ckpt), str(crnn_vocab) if crnn_vocab else None,
            device=device, beam_width=args.beam_width,
        )
        test_ds = IAMLineDataset(
            test_split, predictor.preprocessor, vocabulary=predictor.vocabulary
        )
        test_ds.set_training(False)
        loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=crnn_collate_fn, num_workers=0,
        )
        compute_loss = make_ctc_loss(predictor.vocabulary.blank_index)
        crnn_result = evaluate_crnn(
            predictor.model, loader, predictor.vocabulary, device=device,
            beam_width=args.beam_width, compute_loss=compute_loss,
        )
        metrics["CRNN"] = crnn_result
        save_metrics(crnn_result, str(METRICS_DIR / "crnn_metrics.json"))
        save_predictions(crnn_result["details"], str(PRED_DIR / "crnn_predictions.txt"))
        print(
            f"CRNN  CER={crnn_result['cer']:.4f}  WER={crnn_result['wer']:.4f}  "
            f"val_loss={crnn_result['val_loss']:.4f}  "
            f"avg={crnn_result['avg_inference_time_per_line'] * 1000:.1f} ms/line"
        )

    # ----------------------------------------------------------------- TrOCR
    trocr_result = None
    if trocr_dir is None:
        print("TrOCR: no fine-tuned model dir found -> SKIPPED (results marked unavailable).")
    else:
        print(f"\n=== TrOCR: {trocr_dir} ===")
        from src.models.trocr import TrOCRWrapper

        wrapper = TrOCRWrapper.load_pretrained(trocr_dir, device=device)
        trocr_result = evaluate_trocr(
            wrapper, test_split, device=device, batch_size=args.trocr_batch_size,
            max_length=128, num_beams=args.trocr_beams,
        )
        metrics["TrOCR"] = trocr_result
        save_metrics(trocr_result, str(METRICS_DIR / "trocr_metrics.json"))
        save_predictions(trocr_result["details"], str(PRED_DIR / "trocr_predictions.txt"))
        print(
            f"TrOCR CER={trocr_result['cer']:.4f}  WER={trocr_result['wer']:.4f}  "
            f"avg={trocr_result['avg_inference_time_per_line'] * 1000:.1f} ms/line"
        )

    if not metrics:
        print("\nNo trained checkpoints are available: nothing to compare.")
        return

    # -------------------------------------------------- training information
    crnn_summary = _read_json(CKPT_DIR / "crnn/training_summary.json") or {}
    trocr_summary = _read_json(CKPT_DIR / "trocr/training_summary.json") or {}
    crnn_history = _read_json(CKPT_DIR / "crnn/crnn_history.json") or {}
    trocr_history = _read_json(CKPT_DIR / "trocr/trocr_history.json") or {}

    training_times = {}
    best_val = {}
    if "CRNN" in metrics:
        training_times["CRNN"] = crnn_summary.get("total_time_seconds")
        best_val["CRNN"] = crnn_summary.get("best_val_loss")
    if "TrOCR" in metrics:
        training_times["TrOCR"] = trocr_summary.get("train_runtime")
        best_val["TrOCR"] = trocr_summary.get("best_val_loss")

    comparison = {}
    for name, result in metrics.items():
        comparison[name] = {
            "cer": result["cer"],
            "wer": result["wer"],
            "num_samples": result["num_samples"],
            "avg_inference_time_per_line": result["avg_inference_time_per_line"],
            "total_inference_time": result["total_inference_time"],
            "val_loss": best_val.get(name),
            "training_time_seconds": training_times.get(name),
        }
    with open(METRICS_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # ------------------------------------------------------------- figures
    figures_metrics = {
        name: {"cer": r["cer"], "wer": r["wer"]} for name, r in metrics.items()
    }
    histories = {}
    if crnn_history.get("train_loss"):
        histories["CRNN"] = crnn_history
    if trocr_history.get("train_loss"):
        histories["TrOCR"] = trocr_history

    make_all_figures(
        figures_metrics,
        histories=histories,
        training_times={k: v for k, v in training_times.items() if v},
        figures_dir=str(FIG_DIR),
    )
    plot_time_comparison(
        {name: r["avg_inference_time_per_line"] * 1000 for name, r in metrics.items()},
        str(FIG_DIR / "inference_time.png"),
        title="Average inference time per line",
        ylabel="Milliseconds",
    )

    # ------------------------------------------ side-by-side predictions figure
    n_show = min(6, len(test_split))
    crnn_desc = crnn_result["details"] if crnn_result else []
    trocr_desc = trocr_result["details"] if trocr_result else []
    records = []
    for i in range(n_show):
        records.append(
            {
                "image": test_split[i]["image"],
                "reference": str(test_split[i]["text"]),
                "crnn": crnn_desc[i]["hypothesis"] if crnn_desc else None,
                "trocr": trocr_desc[i]["hypothesis"] if trocr_desc else None,
            }
        )
    plot_predictions_with_images(records, str(FIG_DIR / "prediction_examples.png"))

    print("\n" + "=" * 70)
    print("SUMMARY (IAM-line test split)")
    print("=" * 70)
    for name, values in comparison.items():
        print(
            f"{name:6s} CER={values['cer']:.4f}  WER={values['wer']:.4f}  "
            f"{values['avg_inference_time_per_line'] * 1000:7.1f} ms/line  "
            f"n={values['num_samples']}"
        )
    print(f"\nResults written to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"Elapsed: {time.time() - started:.1f}s")
