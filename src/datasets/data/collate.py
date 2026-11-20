"""Collate function for variable-width CRNN batches.

Pads images to the maximum width in the batch and stacks CTC targets into the
format expected by torch.nn.CTCLoss:
  - targets : 1-D LongTensor of concatenated label indices (sum of lengths)
  - target_lengths : 1-D LongTensor of per-sample label lengths
  - input_lengths : 1-D LongTensor of per-sample CNN output sequence lengths
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_cnn_output_lengths(
    image_widths: list[int],
    img_height: int = 64,
    cnn_downsample_factor: int = 8,
) -> list[int]:
    """Compute the CNN output sequence length for each image width.

    The CRNN in this project uses three stride-2 max-pooling layers, so the
    width is divided by 8. Every convolution uses padding=1 (size preserving),
    therefore the exact number of valid time steps is ``floor(W / 8)``, not
    ``ceil(W / 8)``. Using a value larger than the real number of frames makes
    CTCLoss read past the end of the logits and raises an alignment error, so
    the floor matters.

    ``img_height`` is kept in the signature for callers that only know the
    configured height; it does not affect the result.
    """
    del img_height  # kept for backwards-compatible call sites
    factor = max(1, int(cnn_downsample_factor))
    return [max(1, int(w) // factor) for w in image_widths]


def crnn_collate_fn(
    batch: list[dict],
    pad_value: float = -1.0,
    cnn_downsample_factor: int = 8,
) -> dict:
    """Collate a list of IAMLineDataset samples into a CTC-ready batch.

    Returns
    -------
    images : FloatTensor (B, C, H, W_max) padded with ``pad_value``.
    widths : list[int] true (unpadded) width of every image.
    input_lengths : LongTensor (B,) CNN output sequence length per sample.
    texts : list[str] ground-truth strings (for decoding / metrics).
    targets : LongTensor (sum(target_lengths),) concatenated label indices.
    target_lengths : LongTensor (B,)
    indices : LongTensor (B,) original dataset indices.
    """
    if len(batch) == 0:
        raise ValueError("crnn_collate_fn received an empty batch")

    images = [torch.from_numpy(b["image"]).float() for b in batch]  # (C, H, W)
    widths = [int(im.shape[-1]) for im in images]
    max_width = max(widths)

    # Pad along the width dimension only (right side) and stack.
    padded = []
    for im, w in zip(images, widths):
        if w < max_width:
            im = F.pad(im, (0, max_width - w), mode="constant", value=pad_value)
        padded.append(im)
    images_padded = torch.stack(padded, dim=0)  # (B, C, H, W_max)

    texts = [b["text"] for b in batch]
    target_list = [torch.as_tensor(b["target"], dtype=torch.long) for b in batch]
    target_lengths = torch.tensor(
        [int(b["target_length"]) for b in batch], dtype=torch.long
    )
    indices = torch.tensor([int(b["index"]) for b in batch], dtype=torch.long)

    targets_cat = (
        torch.cat(target_list) if target_list else torch.empty(0, dtype=torch.long)
    )

    input_lengths = torch.tensor(
        compute_cnn_output_lengths(widths, cnn_downsample_factor=cnn_downsample_factor),
        dtype=torch.long,
    )

    return {
        "images": images_padded,
        "widths": widths,
        "input_lengths": input_lengths,
        "texts": texts,
        "targets": targets_cat,
        "target_lengths": target_lengths,
        "indices": indices,
    }
