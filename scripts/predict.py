"""Command-line inference for CRNN and TrOCR.

Usage (VS Code terminal, from HTR_Project/):
    python scripts/predict.py --model crnn --image path/to/line.png \
        --crnn-checkpoint checkpoints/crnn/crnn_best.pt --crnn-vocab checkpoints/crnn/vocab.json
    python scripts/predict.py --model trocr --image path/to/line.png \
        --trocr-dir checkpoints/trocr/final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.inference.predict import load_predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize text from a handwritten line image")
    parser.add_argument("--model", required=True, choices=["crnn", "trocr"])
    parser.add_argument("--image", required=True, help="Path to handwritten line image")
    parser.add_argument("--crnn-checkpoint", default="checkpoints/crnn/crnn_best.pt")
    parser.add_argument("--crnn-vocab", default="checkpoints/crnn/vocab.json")
    parser.add_argument("--trocr-dir", default="checkpoints/trocr/final")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--beam-width", type=int, default=1, help="CRNN beam search width")
    parser.add_argument("--trocr-beams", type=int, default=4)
    args = parser.parse_args()

    if not Path(args.image).is_file():
        parser.error(f"Image not found: {args.image}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "crnn":
        if not Path(args.crnn_checkpoint).is_file():
            parser.error(
                f"CRNN checkpoint not found: {args.crnn_checkpoint}. "
                "Train it with: python scripts/train_crnn.py"
            )
        vocab = args.crnn_vocab if Path(args.crnn_vocab).is_file() else None
        predictor = load_predictor(
            "crnn",
            checkpoint_path=args.crnn_checkpoint,
            vocab_path=vocab,
            device=device,
            beam_width=args.beam_width,
        )
    else:
        if not Path(args.trocr_dir).is_dir():
            parser.error(
                f"TrOCR model dir not found: {args.trocr_dir}. "
                "Fine-tune it with: python scripts/train_trocr.py"
            )
        predictor = load_predictor(
            "trocr",
            model_dir=args.trocr_dir,
            device=device,
            max_length=128,
            num_beams=args.trocr_beams,
        )

    result = predictor.predict(args.image)

    print("-" * 60)
    print(f"Model   : {result.model_type.upper()}")
    print(f"Image   : {args.image}")
    print(f"Time    : {result.inference_time * 1000:.1f} ms")
    print(f"Text    : {result.text}")
    print("-" * 60)


if __name__ == "__main__":
    main()
