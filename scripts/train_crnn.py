"""Train CRNN + BiLSTM + CTC on IAM line-level data.

Usage (VS Code terminal, from HTR_Project/):
    python scripts/train_crnn.py --config configs/crnn_config.yaml --epochs 20
    python scripts/train_crnn.py --epochs 1 --max-train 128 --max-val 32   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.collate import crnn_collate_fn
from src.data.dataset import build_iam_datasets, load_iam_splits
from src.data.preprocessing import CRNNPreprocessConfig, CRNNPreprocessor
from src.data.sampling import WidthBucketBatchSampler
from src.data.vocabulary import Vocabulary
from src.models.crnn import CRNN
from src.training.losses import make_ctc_loss
from src.training.trainer import BaseTrainer


def build_vocabulary(args, model_cfg) -> Vocabulary:
    """Build the character vocabulary from the *training* split only."""
    train_split = load_iam_splits()["train"]
    texts = train_split["text"]
    if args.max_train:
        texts = texts[: args.max_train]
    vocab = Vocabulary(
        blank_token=model_cfg.get("blank_token", "<blank>"),
        unk_token=" unk",
        use_unk=True,
    ).build_from_texts(texts)
    print(f"Vocabulary size: {vocab.num_classes} (blank=0, unk=1, chars={len(vocab.chars)})")
    return vocab


def dataset_widths(dataset, img_height: int, cache_path: str) -> list[int]:
    """Per-sample width after aspect-preserving resize (cached to JSON)."""
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) == len(dataset):
            return cached

    widths = []
    for i in range(len(dataset)):
        image = dataset.hf_dataset[i][dataset.image_column]
        w, h = image.size
        widths.append(int(round(w * img_height / max(1, h))))
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(widths, f)
    return widths


def padding_saving(widths, batches) -> tuple[float, float]:
    """Mean sample width vs mean padded batch width."""
    mean_w = sum(widths) / max(1, len(widths))
    total = sum(len(b) * max(widths[i] for i in b) for b in batches)
    n = sum(len(b) for b in batches)
    return mean_w, total / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRNN + BiLSTM + CTC on IAM-line")
    parser.add_argument("--config", default="configs/crnn_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-train", type=int, default=None, help="cap train samples (smoke test)")
    parser.add_argument("--max-val", type=int, default=None, help="cap val samples (smoke test)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    parser.add_argument("--output-dir", default="checkpoints/crnn")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 = in-process, most stable on Windows)")
    parser.add_argument("--no-amp", action="store_true",
                        help="disable mixed precision (automatic on CUDA)")
    parser.add_argument("--no-bucket", action="store_true",
                        help="disable width-bucketed batching (random order instead)")
    parser.add_argument("--bucket-batches", type=int, default=50,
                        help="batches per width bucket (larger = tighter widths)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug_cfg = cfg.get("augmentation", {})

    # Reproducibility (config seed is 42).
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        # Variable image widths would make cudnn autotuning re-benchmark every
        # new shape; keep the default deterministic algorithm selection.
        torch.backends.cudnn.benchmark = False
        # Ampere TF32 for fp32 matmuls (LSTM); ~5-10% faster, negligible error.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ``scale`` is a (min, max) range; accept a single float for convenience.
    scale_cfg = aug_cfg.get("scale", [0.95, 1.05])
    if isinstance(scale_cfg, (int, float)):
        scale_cfg = (float(scale_cfg), 1.0 + (1.0 - float(scale_cfg)))
    else:
        scale_cfg = tuple(scale_cfg)

    prep_cfg = CRNNPreprocessConfig(
        img_height=model_cfg.get("img_height", 64),
        enabled=aug_cfg.get("enabled", True),
        rotation_degrees=aug_cfg.get("rotation_degrees", 2.0),
        translate=aug_cfg.get("translate", 0.02),
        scale=scale_cfg,
        brightness=aug_cfg.get("brightness", 0.2),
        contrast=aug_cfg.get("contrast", 0.2),
    )
    preprocessor = CRNNPreprocessor(prep_cfg)

    vocab = build_vocabulary(args, model_cfg)
    vocab.save(f"{args.output_dir}/vocab.json")

    train_ds, val_ds, _ = build_iam_datasets(
        preprocessor,
        vocabulary=vocab,
        max_train=args.max_train,
        max_val=args.max_val,
    )
    batch_size = args.batch_size or train_cfg.get("batch_size", 16)
    cache_dir = args.output_dir
    train_sampler = val_sampler = None
    if not args.no_bucket:
        train_widths = dataset_widths(
            train_ds,
            prep_cfg.img_height,
            f"{cache_dir}/widths_train_{len(train_ds)}_{prep_cfg.img_height}.json",
        )
        val_widths = dataset_widths(
            val_ds,
            prep_cfg.img_height,
            f"{cache_dir}/widths_val_{len(val_ds)}_{prep_cfg.img_height}.json",
        )
        train_sampler = WidthBucketBatchSampler(
            train_widths, batch_size, bucket_size_batches=args.bucket_batches
        )
        val_sampler = WidthBucketBatchSampler(
            val_widths, batch_size, bucket_size_batches=args.bucket_batches, shuffle=False
        )
        mean_w, padded_w = padding_saving(train_widths, train_sampler._batches())
        print(
            f"Width bucketing: mean {mean_w:.0f}px | avg padded {padded_w:.0f}px "
            f"({100 * (1 - mean_w / max(1.0, padded_w)):.0f}% padding saved)"
        )

    loader_kwargs = {
        "collate_fn": crnn_collate_fn,
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
    }
    if train_sampler is None:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **loader_kwargs
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **loader_kwargs)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model_kwargs = {
        "img_height": model_cfg.get("img_height", 64),
        "num_classes": vocab.num_classes,
        "rnn_hidden_size": model_cfg.get("rnn_hidden_size", 256),
        "rnn_num_layers": model_cfg.get("rnn_num_layers", 2),
        "dropout": model_cfg.get("dropout", 0.3),
    }
    model = CRNN(**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CRNN parameters: {n_params / 1e6:.2f}M")

    use_amp = device == "cuda" and not args.no_amp
    print(f"Mixed precision: {'fp16' if use_amp else 'off'}")
    compute_loss = make_ctc_loss(blank_index=vocab.blank_index, amp=use_amp)

    trainer = BaseTrainer(
        model=model,
        device=device,
        learning_rate=float(args.lr or train_cfg.get("learning_rate", 0.001)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0001)),
        grad_clip=train_cfg.get("grad_clip", 5.0),
        scheduler_mode=train_cfg.get("scheduler", "reduce_on_plateau"),
        scheduler_patience=train_cfg.get("scheduler_patience", 5),
        scheduler_factor=train_cfg.get("scheduler_factor", 0.5),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 10),
        checkpoint_dir=args.output_dir,
        run_name="crnn",
        use_amp=use_amp,
        extra_state={
            "model": model_kwargs,
            "vocabulary": {
                "blank_token": vocab.blank_token,
                "unk_token": vocab.unk_token,
                "use_unk": vocab.use_unk,
                "chars": vocab.chars,
            },
            "dataset": "Teklia/IAM-line",
            "train_args": vars(args),
            "amp": use_amp,
            "bucket_batches": None if args.no_bucket else args.bucket_batches,
        },
    )

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume}, starting at epoch {start_epoch + 1}")

    epochs = args.epochs or train_cfg.get("epochs", 50)
    started = time.time()
    result = trainer.fit(
        train_loader, val_loader, compute_loss, epochs=epochs, start_epoch=start_epoch
    )

    summary = {
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch + 1,
        "total_time_seconds": result.total_time,
        "epochs_requested": epochs,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "device": str(trainer.device),
        "parameters": n_params,
        "wall_clock_seconds": time.time() - started,
        "amp": use_amp,
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available()
            else 0.0
        ),
    }
    import json

    with open(f"{args.output_dir}/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Training summary:", summary)
    print(f"Checkpoints written to {args.output_dir}")


if __name__ == "__main__":
    main()
