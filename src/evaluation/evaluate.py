"""Evaluate CRNN and TrOCR on the same IAM test samples.

Both evaluators return the same dictionary shape so the comparison page and the
Streamlit app can consume them without special cases:

    {
      "model", "cer", "wer", "num_samples",
      "total_inference_time", "avg_inference_time_per_line",
      "references", "hypotheses", "details"
    }

Only inference/generation time is measured (preprocessing is excluded).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from .metrics import compute_cer_wer


def evaluate_crnn(
    model: torch.nn.Module,
    loader: DataLoader,
    vocabulary,
    device: str = "cpu",
    beam_width: int = 1,
    compute_loss=None,
) -> Dict:
    """Evaluate CRNN on a DataLoader built with ``crnn_collate_fn``."""
    from ..models.decoder import beam_search_decode, greedy_decode

    model.eval()
    model.to(device)

    references: List[str] = []
    hypotheses: List[str] = []
    total_time = 0.0
    val_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            texts = batch["texts"]

            if compute_loss is not None:
                val_loss += float(compute_loss(model, batch, device, False))
                n_batches += 1

            t0 = time.perf_counter()
            logits = model(images)
            t1 = time.perf_counter()
            total_time += t1 - t0

            log_probs = torch.log_softmax(logits, dim=-1).detach()
            if beam_width > 1:
                pred_indices = beam_search_decode(
                    log_probs, beam_width=beam_width, blank_index=vocabulary.blank_index
                )
            else:
                pred_indices = greedy_decode(
                    log_probs, blank_index=vocabulary.blank_index
                )

            hypotheses.extend(vocabulary.decode_batch(pred_indices))
            references.extend(str(t) for t in texts)

    cer_score, wer_score, details = compute_cer_wer(references, hypotheses)
    n_samples = len(references)
    return {
        "model": "CRNN",
        "cer": cer_score,
        "wer": wer_score,
        "num_samples": n_samples,
        "total_inference_time": total_time,
        "avg_inference_time_per_line": total_time / n_samples if n_samples else 0.0,
        "val_loss": val_loss / n_batches if n_batches else None,
        "references": references,
        "hypotheses": hypotheses,
        "details": details,
    }


def evaluate_trocr(
    wrapper,
    hf_split,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 128,
    num_beams: int = 4,
    max_samples: Optional[int] = None,
) -> Dict:
    """Evaluate a TrOCR wrapper on an IAM-line split (list of image/text rows)."""
    wrapper.model.eval()
    if max_samples is not None:
        hf_split = hf_split.select(range(min(max_samples, len(hf_split))))

    images = hf_split["image"]
    references = [str(t) for t in hf_split["text"]]
    hypotheses: List[str] = []
    total_time = 0.0

    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        pixel_values = wrapper.processor(images=batch_images, return_tensors="pt")[
            "pixel_values"
        ]
        t0 = time.perf_counter()
        hypotheses.extend(
            wrapper.generate(
                pixel_values=pixel_values, max_length=max_length, num_beams=num_beams
            )
        )
        total_time += time.perf_counter() - t0

    if len(hypotheses) != len(references):
        raise RuntimeError(
            f"TrOCR produced {len(hypotheses)} predictions for {len(references)} samples"
        )

    cer_score, wer_score, details = compute_cer_wer(references, hypotheses)
    n_samples = len(references)
    return {
        "model": "TrOCR",
        "cer": cer_score,
        "wer": wer_score,
        "num_samples": n_samples,
        "total_inference_time": total_time,
        "avg_inference_time_per_line": total_time / n_samples if n_samples else 0.0,
        "val_loss": None,
        "references": references,
        "hypotheses": hypotheses,
        "details": details,
    }


def save_metrics(metrics: Dict, path: str) -> None:
    """Persist metrics without the large per-sample arrays."""
    clean = {
        k: v
        for k, v in metrics.items()
        if k not in ("references", "hypotheses", "details")
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)


def save_predictions(details: List[Dict], path: str, max_examples: Optional[int] = None) -> None:
    """Save ground-truth vs prediction pairs as TSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = details if max_examples is None else details[:max_examples]
    with open(path, "w", encoding="utf-8") as f:
        f.write("index\tground_truth\tprediction\tCER\tWER\n")
        for i, row in enumerate(rows):
            ref = str(row["reference"]).replace("\t", " ")
            hyp = str(row["hypothesis"]).replace("\t", " ")
            f.write(f"{i}\t{ref}\t{hyp}\t{row['cer']:.4f}\t{row['wer']:.4f}\n")
