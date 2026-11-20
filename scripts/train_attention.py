"""Train the CNN + BiLSTM + Bahdanau attention + autoregressive decoder model.

This is the project's third architecture (``src/models/attention_htr.py``). It
shares the dataset, preprocessing, width bucketing, trainer scaffolding and
checkpoint conventions with the CRNN pipeline, but it is trained with a masked
token-level cross-entropy instead of CTC.

Usage (VS Code terminal, from HTR_Project/):
    python scripts/train_attention.py --config configs/attention_config.yaml
    python scripts/train_attention.py --epochs 30 --batch-size 16 --device cuda
    python scripts/train_attention.py --epochs 2 --max-train 128 --max-val 32   # smoke test
    python scripts/train_attention.py --resume checkpoints/attention/attention_latest.pt

The full 30-epoch run is deliberately not started automatically; run the command
above once the implementation has been reviewed and tested.
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

from scripts.train_crnn import dataset_widths, padding_saving
from src.data.dataset import build_iam_datasets, load_iam_splits
from src.data.preprocessing import CRNNPreprocessConfig, CRNNPreprocessor
from src.data.sampling import WidthBucketBatchSampler
from src.datasets.data.attention import AttentionCollator
from src.datasets.data.attention_vocab import AttentionVocabulary
from src.models.attention_htr import AttentionHTR
from src.training.attention_trainer import AttentionTrainer
from src.training.losses import make_attention_loss

SEED = 42


def build_vocabulary(max_train: int | None) -> AttentionVocabulary:
    """Character set from the TRAINING split only (no leakage)."""
    texts = load_iam_splits()["train"]["text"]
    if max_train:
        texts = texts[:max_train]
    vocab = AttentionVocabulary.build_from_texts(texts)
    print(
        f"Vocabulary: {vocab.num_classes} classes "
        f"(PAD=0 SOS=1 EOS=2 UNK=3, {len(vocab.chars)} characters)"
    )
    return vocab


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` but never silently fall back when CUDA was requested."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested with --device cuda but torch.cuda.is_available() is False.\n"
            "Install the CUDA build of PyTorch, or pass --device cpu explicitly:\n"
            "  python -m pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu124"
        )
    if requested == "cpu" and torch.cuda.is_available():
        print("Note: CUDA is available but --device cpu was requested; using CPU.")
    return requested


def scale_range(cfg: dict) -> tuple[float, float]:
    """YAML may express ``scale`` as a range or a single float."""
    value = cfg.get("scale", [0.95, 1.05])
    if isinstance(value, (int, float)):
        return (float(value), 1.0 + (1.0 - float(value)))
    return tuple(value)  # type: ignore[return-value]


def show_predictions(trainer, val_loader, vocabulary, device: str, n: int = 5) -> None:
    """Print a few validation predictions: the smoke test of readability."""
    if n <= 0:
        return
    trainer.model.eval()
    shown = 0
    print("\nValidation predictions (greedy decoding):")
    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            lengths = batch["encoder_lengths"].to(device)
            ids = trainer.model.greedy_decode(images, lengths, max_length=128)
            for reference, prediction in zip(batch["texts"], vocabulary.decode_batch(ids)):
                print(f"  GT  : {reference}")
                print(f"  PRED: {prediction}")
                shown += 1
                if shown >= n:
                    return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNN + BiLSTM + attention decoder on IAM-line"
    )
    parser.add_argument("--config", default="configs/attention_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accum-steps", type=int, default=None,
                        help="gradient accumulation steps (effective batch = batch-size * steps)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--teacher-forcing", type=float, default=None,
                        help="constant teacher-forcing ratio (disables annealing)")
    parser.add_argument("--tf-start", type=float, default=None)
    parser.add_argument("--tf-end", type=float, default=None)
    parser.add_argument("--tf-anneal-epochs", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None,
                        help="maximum decoder length including <EOS>")
    parser.add_argument("--max-train", type=int, default=None, help="cap train samples (smoke test)")
    parser.add_argument("--max-val", type=int, default=None, help="cap val samples (smoke test)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    parser.add_argument("--output-dir", default="checkpoints/attention")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--no-bucket", action="store_true",
                        help="disable width-bucketed batching (random order instead)")
    parser.add_argument("--bucket-batches", type=int, default=50)
    parser.add_argument("--show-predictions", type=int, default=5)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    aug_cfg = cfg.get("augmentation", {})

    # ------------------------------------------------------------ deterministic
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = resolve_device(args.device)
    print(f"Device: {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1e9:.2f} GB VRAM")

    img_height = int(model_cfg.get("img_height", 64))
    max_length = int(args.max_length or model_cfg.get("max_length", 128))

    prep_cfg = CRNNPreprocessConfig(
        img_height=img_height,
        enabled=aug_cfg.get("enabled", True),
        rotation_degrees=aug_cfg.get("rotation_degrees", 2.0),
        translate=aug_cfg.get("translate", 0.02),
        scale=scale_range(aug_cfg),
        brightness=aug_cfg.get("brightness", 0.2),
        contrast=aug_cfg.get("contrast", 0.2),
    )
    preprocessor = CRNNPreprocessor(prep_cfg)

    vocab = build_vocabulary(args.max_train)
    os.makedirs(args.output_dir, exist_ok=True)
    vocab.save(os.path.join(args.output_dir, "attention_vocab.json"))

    # The attention model does not use CTC targets, so no vocabulary is passed
    # to the dataset; the collator tokenises the raw text instead.
    train_ds, val_ds, _ = build_iam_datasets(
        preprocessor,
        vocabulary=None,
        max_train=args.max_train,
        max_val=args.max_val,
    )

    batch_size = int(args.batch_size or train_cfg.get("batch_size", 16))
    accum_steps = int(args.accum_steps or train_cfg.get("accum_steps", 1))
    collator = AttentionCollator(vocab, max_length=max_length)

    train_sampler = val_sampler = None
    if not args.no_bucket:
        train_widths = dataset_widths(
            train_ds, img_height,
            f"{args.output_dir}/widths_train_{len(train_ds)}_{img_height}.json",
        )
        val_widths = dataset_widths(
            val_ds, img_height,
            f"{args.output_dir}/widths_val_{len(val_ds)}_{img_height}.json",
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
        "collate_fn": collator,
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
    print(f"Batch size: {batch_size} | gradient accumulation: {accum_steps} "
          f"(effective {batch_size * accum_steps})")

    # ------------------------------------------------------------------ model
    model_kwargs = {
        "vocab_size": vocab.num_classes,
        "img_height": img_height,
        "cnn_dropout": float(model_cfg.get("cnn_dropout", 0.1)),
        "encoder_hidden": int(model_cfg.get("encoder_hidden", 128)),
        "encoder_layers": int(model_cfg.get("encoder_layers", 1)),
        "decoder_hidden": int(model_cfg.get("decoder_hidden", 128)),
        "decoder_layers": int(model_cfg.get("decoder_layers", 1)),
        "embed_dim": int(model_cfg.get("embed_dim", 128)),
        "attn_dim": int(model_cfg.get("attn_dim", 128)),
        "dropout": float(model_cfg.get("dropout", 0.2)),
        "pad_index": vocab.pad_index,
        "sos_index": vocab.sos_index,
        "eos_index": vocab.eos_index,
        "unk_index": vocab.unk_index,
    }
    model = AttentionHTR(**model_kwargs)
    n_params = model.num_parameters()
    print(f"AttentionHTR parameters: {n_params / 1e6:.3f}M")
    print(
        f"CNN: {img_height}px height -> {model.cnn.out_height} rows, "
        f"{model.cnn.out_channels} channels ({model.cnn.feature_dim} features/step), "
        f"T = floor(W/{model.cnn.width_downsample})"
    )

    use_amp = device == "cuda" and not args.no_amp
    print(f"Mixed precision: {'FP16 CNN stack (LSTM/decoder in FP32)' if use_amp else 'off'}")
    compute_loss = make_attention_loss(
        ignore_index=vocab.pad_index, amp=use_amp
    )

    tf_start = float(
        args.tf_start
        if args.tf_start is not None
        else train_cfg.get("teacher_forcing_start", 0.9)
    )
    tf_end = float(
        args.tf_end if args.tf_end is not None else train_cfg.get("teacher_forcing_end", 0.5)
    )
    if args.teacher_forcing is not None:
        # A single constant ratio, as suggested in the task specification.
        tf_start = tf_end = float(args.teacher_forcing)
    tf_anneal = args.tf_anneal_epochs
    if tf_anneal is None and args.teacher_forcing is None:
        tf_anneal = train_cfg.get("teacher_forcing_anneal_epochs", 10)
    print(f"Teacher forcing: {tf_start:.2f} -> {tf_end:.2f} over {tf_anneal or 'n/a'} epochs")

    epochs = int(args.epochs or train_cfg.get("epochs", 30))

    trainer = AttentionTrainer(
        model=model,
        device=device,
        accum_steps=accum_steps,
        teacher_forcing_start=tf_start,
        teacher_forcing_end=tf_end,
        teacher_forcing_anneal_epochs=tf_anneal if tf_anneal else None,
        learning_rate=float(args.lr or train_cfg.get("learning_rate", 0.0003)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0001)),
        grad_clip=train_cfg.get("gradient_clip", train_cfg.get("grad_clip", 1.0)),
        scheduler_mode=train_cfg.get("scheduler", "reduce_on_plateau"),
        scheduler_patience=int(train_cfg.get("scheduler_patience", 3)),
        scheduler_factor=float(train_cfg.get("scheduler_factor", 0.5)),
        early_stopping_patience=int(train_cfg.get("early_stopping_patience", 5)),
        checkpoint_dir=args.output_dir,
        run_name="attention",
        use_amp=use_amp,
        extra_state={
            "model": model.config_dict(),
            "vocabulary": vocab.to_dict(),
            "dataset": "Teklia/IAM-line",
            "max_length": max_length,
            "train_args": vars(args),
            "amp": use_amp,
            "teacher_forcing": {
                "start": tf_start,
                "end": tf_end,
                "anneal_epochs": tf_anneal,
            },
            "batch_size": batch_size,
            "accum_steps": accum_steps,
        },
    )

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        trainer.set_start_epoch(start_epoch)
        print(f"Resumed from {args.resume}, starting at epoch {start_epoch + 1}")

    started = time.time()
    result = trainer.fit(
        train_loader, val_loader, compute_loss, epochs=epochs, start_epoch=start_epoch
    )

    summary = {
        "model": "AttentionHTR",
        "architecture": "CNN + BiLSTM + Bahdanau attention + LSTM decoder",
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch + 1,
        "total_time_seconds": result.total_time,
        "epochs_requested": epochs,
        "epochs_run": len(result.history.get("train_loss", [])),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "batch_size": batch_size,
        "accum_steps": accum_steps,
        "learning_rate": float(args.lr or train_cfg.get("learning_rate", 0.0003)),
        "teacher_forcing": {
            "start": tf_start,
            "end": tf_end,
            "anneal_epochs": tf_anneal,
        },
        "max_length": max_length,
        "device": str(trainer.device),
        "parameters": n_params,
        "wall_clock_seconds": time.time() - started,
        "amp": use_amp,
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        ),
        "checkpoint": os.path.join(args.output_dir, "attention_best.pt"),
    }
    with open(f"{args.output_dir}/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining summary:")
    for key in (
        "best_val_loss", "best_epoch", "epochs_run", "total_time_seconds",
        "parameters", "peak_vram_gb", "amp",
    ):
        print(f"  {key}: {summary[key]}")

    show_predictions(trainer, val_loader, vocab, device, n=args.show_predictions)
    print(f"\nCheckpoints written to {args.output_dir}")


if __name__ == "__main__":
    main()
