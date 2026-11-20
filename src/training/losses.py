"""Loss functions shared by training and evaluation."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

_WARNED = {"alignment": False}


def make_ctc_loss(blank_index: int = 0, zero_infinity: bool = True, amp: bool = False):
    """Build a compute_loss(model, batch, device, train) callable for CTCLoss.

    The callable validates the CTC alignment lengths and warns loudly when a
    target cannot fit into the available frames instead of failing silently.
    """

    def compute_loss(model, batch, device, train: bool = False):
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        input_lengths = batch.get("input_lengths")
        if input_lengths is None:
            input_lengths = model.get_output_lengths(batch["widths"])
            input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        input_lengths = input_lengths.to(device)

        if hasattr(model, "set_fp16_cnn"):
            # FP16 only for the convolutional stack; LSTM + CTC stay FP32.
            model.set_fp16_cnn(bool(amp) and images.is_cuda)
        logits = model(images)  # (T, B, C)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        T = log_probs.shape[0]
        if int(input_lengths.max()) > T:
            raise ValueError(
                f"CTC input length {int(input_lengths.max())} exceeds the {T} frames "
                "produced by the CNN. The length formula and the CNN are out of sync."
            )

        if (target_lengths > input_lengths).any() and not _WARNED["alignment"]:
            warnings.warn(
                "At least one CTC target is longer than its input sequence "
                "(target_length > input_length). Those samples will be ignored "
                "by CTCLoss (zero_infinity=True); check the image height/width "
                "settings if this happens often.",
                RuntimeWarning,
            )
            _WARNED["alignment"] = True

        loss = nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank_index,
            reduction="mean",
            zero_infinity=zero_infinity,
        )
        return loss

    return compute_loss
