"""Fine-tune pretrained TrOCR on IAM line-level data.

Usage (VS Code terminal, from HTR_Project/):
    python scripts/train_trocr.py --config configs/trocr_config.yaml --epochs 3
    python scripts/train_trocr.py --epochs 1 --max-train 32 --max-val 8   # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
)

from src.data.dataset import TrOCRCollator, TrOCRDataset, load_iam_splits
from src.models.trocr import (
    _read_decoder_start,
    build_processor,
    materialise_model_dir,
)


def _history_from_logs(log_history: list[dict]) -> dict:
    """Convert Trainer log entries into train_loss / val_loss lists."""
    train_loss, val_loss = [], []
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_loss.append(float(entry["loss"]))
        if "eval_loss" in entry:
            val_loss.append(float(entry["eval_loss"]))
    return {"train_loss": train_loss, "val_loss": val_loss}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on IAM-line")
    parser.add_argument("--config", default="configs/trocr_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", default=None, help="Trainer checkpoint dir OR fine-tuned model dir")
    parser.add_argument("--output-dir", default="checkpoints/trocr")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="trade compute for VRAM (recommended on 4 GB GPUs)")
    parser.add_argument("--no-fp16", action="store_true", help="disable mixed precision on CUDA")
    parser.add_argument("--adafactor", action="store_true", help="use Adafactor (less optimizer VRAM)")
    parser.add_argument("--eval-subset", type=int, default=None,
                        help="cap validation samples used during training")
    parser.add_argument("--generation-max-length", type=int, default=64,
                        help="max generated tokens for validation metrics")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    gen_cfg = cfg.get("generation", {})

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if args.device == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False")
    print(f"Device: {'cuda' if use_cuda else 'cpu'}")

    checkpoint = model_cfg.get("checkpoint", "microsoft/trocr-small-handwritten")
    max_label_length = gen_cfg.get("max_length", 128)

    processor = build_processor(checkpoint)
    # The published checkpoint is a legacy .bin file; load a safetensors copy.
    model_dir = materialise_model_dir(checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    if args.resume and Path(args.resume).is_dir() and (
        Path(args.resume) / "config.json"
    ).is_file():
        print(f"Initialising weights from fine-tuned model dir: {args.resume}")
        model = VisionEncoderDecoderModel.from_pretrained(
            materialise_model_dir(args.resume)
        )

    # TrOCR decodes from </s> (id 2) as published; never replace it with <s>.
    decoder_start = _read_decoder_start(model, processor.tokenizer.eos_token_id)
    model.config.decoder_start_token_id = decoder_start
    model.generation_config.decoder_start_token_id = decoder_start
    model.config.decoder.eos_token_id = processor.tokenizer.eos_token_id
    model.config.decoder.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder.bos_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.decoder.max_length = max_label_length
    model.generation_config.max_length = max_label_length
    # transformers 5 rejects early_stopping with num_beams=1 when saving the
    # generation config; beam search early stopping is applied at generate().
    model.generation_config.early_stopping = False

    hf = load_iam_splits()
    train_split, val_split = hf["train"], hf["validation"]
    if args.max_train:
        train_split = train_split.select(range(min(args.max_train, len(train_split))))
    if args.max_val:
        val_split = val_split.select(range(min(args.max_val, len(val_split))))
    if args.eval_subset:
        val_split = val_split.select(range(min(args.eval_subset, len(val_split))))

    if args.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("Gradient checkpointing: enabled")

    train_ds = TrOCRDataset(train_split, processor, max_length=max_label_length)
    val_ds = TrOCRDataset(val_split, processor, max_length=max_label_length)
    collator = TrOCRCollator(processor.tokenizer, processor.tokenizer.pad_token_id)

    def compute_metrics(pred) -> dict:
        """Character error rate on validation (free-running generation)."""
        labels = pred.label_ids
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        labels = labels.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        hyps = processor.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.batch_decode(labels, skip_special_tokens=True)
        from src.evaluation.metrics import compute_cer_wer

        cer_score, wer_score, _ = compute_cer_wer(refs, hyps)
        return {"cer": cer_score, "wer": wer_score}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size or train_cfg.get("batch_size", 8),
        per_device_eval_batch_size=args.batch_size or train_cfg.get("batch_size", 8),
        gradient_accumulation_steps=args.grad_accum or 1,
        num_train_epochs=args.epochs or train_cfg.get("epochs", 10),
        learning_rate=float(args.lr or train_cfg.get("learning_rate", 0.00005)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_steps=int(train_cfg.get("warmup_steps", 500)),
        label_smoothing_factor=float(train_cfg.get("label_smoothing", 0.0)),
        max_grad_norm=float(train_cfg.get("grad_clip", 1.0)),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=1,
        fp16=(use_cuda and not args.no_fp16),
        optim="adafactor" if args.adafactor else "adamw_torch",
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_num_workers=args.num_workers,
        seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    resume_from = None
    if args.resume and (Path(args.resume) / "trainer_state.json").is_file():
        resume_from = args.resume
        print(f"Resuming Trainer state from {resume_from}")

    started = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    wall_clock = time.time() - started
    print("Train metrics:", train_result.metrics)

    eval_metrics = trainer.evaluate()
    print("Eval metrics:", eval_metrics)

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Saved fine-tuned TrOCR model + processor to {final_dir}")

    history = _history_from_logs(trainer.state.log_history)
    summary = {
        "wall_clock_seconds": wall_clock,
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_loss": train_result.metrics.get("train_loss"),
        "best_val_loss": eval_metrics.get("eval_loss"),
        "eval_cer": eval_metrics.get("eval_cer"),
        "eval_wer": eval_metrics.get("eval_wer"),
        "epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "grad_accum": training_args.gradient_accumulation_steps,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "checkpoint": checkpoint,
        "fp16": bool(use_cuda and not args.no_fp16),
        "grad_checkpointing": bool(args.grad_checkpointing),
        "optimizer": "adafactor" if args.adafactor else "adamw_torch",
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        ),
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(Path(args.output_dir) / "trocr_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("Training summary:", summary)


if __name__ == "__main__":
    main()
