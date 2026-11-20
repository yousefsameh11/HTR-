"""Shared training entry point for the HTR project.

Dispatches to the CRNN or TrOCR training pipeline from a single command:

    python src/train.py crnn --epochs 50
    python src/train.py trocr --epochs 10

For the full list of options, run with --help on each subcommand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a handwritten text recognition model (CRNN or TrOCR)."
    )
    sub = parser.add_subparsers(dest="model", required=True)

    # ---- CRNN ----------------------------------------------------------------
    crnn = sub.add_parser("crnn", help="Train CRNN + BiLSTM + CTC")
    crnn.add_argument("--config", default="configs/crnn_config.yaml")
    crnn.add_argument("--epochs", type=int, default=None)
    crnn.add_argument("--batch-size", type=int, default=None)
    crnn.add_argument("--lr", type=float, default=None)
    crnn.add_argument("--max-train", type=int, default=None, help="cap train samples (smoke test)")
    crnn.add_argument("--max-val", type=int, default=None, help="cap val samples (smoke test)")
    crnn.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    crnn.add_argument("--resume", default=None, help="checkpoint path to resume from")
    crnn.add_argument("--output-dir", default="checkpoints/crnn")
    crnn.add_argument("--num-workers", type=int, default=0)
    crnn.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    crnn.add_argument("--no-bucket", action="store_true", help="disable width-bucketed batching")
    crnn.add_argument("--bucket-batches", type=int, default=50)

    # ---- TrOCR --------------------------------------------------------------
    trocr = sub.add_parser("trocr", help="Fine-tune pretrained TrOCR")
    trocr.add_argument("--config", default="configs/trocr_config.yaml")
    trocr.add_argument("--epochs", type=int, default=None)
    trocr.add_argument("--batch-size", type=int, default=None)
    trocr.add_argument("--grad-accum", type=int, default=None)
    trocr.add_argument("--lr", type=float, default=None)
    trocr.add_argument("--max-train", type=int, default=None)
    trocr.add_argument("--max-val", type=int, default=None)
    trocr.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    trocr.add_argument("--resume", default=None, help="Trainer checkpoint dir OR fine-tuned model dir")
    trocr.add_argument("--output-dir", default="checkpoints/trocr")
    trocr.add_argument("--num-workers", type=int, default=0)
    trocr.add_argument("--grad-checkpointing", action="store_true")
    trocr.add_argument("--no-fp16", action="store_true", help="disable mixed precision on CUDA")
    trocr.add_argument("--adafactor", action="store_true", help="use Adafactor optimizer")
    trocr.add_argument("--eval-subset", type=int, default=None)
    trocr.add_argument("--generation-max-length", type=int, default=64)

    args = parser.parse_args()

    if args.model == "crnn":
        from scripts.train_crnn import main as crnn_main

        # scripts.train_crnn uses its own argparse; re-encode the relevant flags.
        crnn_main()
    else:
        from scripts.train_trocr import main as trocr_main

        trocr_main()


if __name__ == "__main__":
    main()
