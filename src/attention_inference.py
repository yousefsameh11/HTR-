"""Single-image inference for the CNN + BiLSTM + attention decoder model.

Loads ``checkpoints/attention/attention_best.pt`` (which embeds the model
configuration and the character vocabulary), applies the *same* preprocessing
used during training, decodes greedily until ``<EOS>`` and reports the
model-only inference time.

Used by ``scripts/predict.py --model attention``, ``scripts/evaluate_attention.py``
and the Streamlit application.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.data.preprocessing import CRNNPreprocessConfig, CRNNPreprocessor
from src.datasets.data.attention_vocab import AttentionVocabulary
from src.models.attention_htr import AttentionHTR

from .predict import PredictionResult


def _ensure_pil(image, mode: Optional[str] = None) -> Image.Image:
    """Accept a path, PIL image or 2-D numpy array; reject anything unusable."""
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim != 2:
            raise ValueError(
                f"Expected a 2-D grayscale array for a line image, got shape {array.shape}."
            )
        image = Image.fromarray(array.astype(np.uint8))
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")
    if image.width < 2 or image.height < 2:
        raise ValueError(f"Image is too small to recognise: {image.size[0]}x{image.size[1]} px")
    if mode is not None and image.mode != mode:
        image = image.convert(mode)
    return image


class AttentionInference:
    """Checkpoint-backed predictor for :class:`AttentionHTR`."""

    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: Optional[str] = None,
        device: str = "auto",
        max_length: int = 128,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.checkpoint_path = str(checkpoint_path)
        self.max_length = int(max_length)

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Attention HTR checkpoint not found: {self.checkpoint_path}")

        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(state, dict) or "model_state_dict" not in state:
            raise ValueError(
                f"{self.checkpoint_path} is not a training checkpoint (no model_state_dict)."
            )

        config = state.get("config") or {}
        model_cfg = dict(config.get("model") or {})
        if not model_cfg:
            raise ValueError(
                f"{self.checkpoint_path} does not embed its model configuration; "
                "retrain with scripts/train_attention.py."
            )

        if vocab_path is not None:
            self.vocabulary = AttentionVocabulary.load(vocab_path)
        elif config.get("vocabulary"):
            self.vocabulary = AttentionVocabulary.from_dict(config["vocabulary"])
        else:
            raise ValueError(
                "No vocabulary available: pass vocab_path or use a checkpoint that stores "
                "config['vocabulary'] (written by scripts/train_attention.py)."
            )

        model_cfg["vocab_size"] = self.vocabulary.num_classes
        self.model = AttentionHTR(**model_cfg)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.img_height = int(model_cfg.get("img_height", 64))
        self.preprocessor = CRNNPreprocessor(CRNNPreprocessConfig(img_height=self.img_height))
        self.epoch = state.get("epoch")

    @torch.no_grad()
    def predict(self, image) -> PredictionResult:
        pil = _ensure_pil(image, "L")
        array = self.preprocessor(pil, training=False)              # (1, H, W)
        tensor = torch.from_numpy(array).unsqueeze(0).to(self.device)  # (1, 1, H, W)
        lengths = self.model.encoder_lengths_from_widths([array.shape[-1]]).to(self.device)

        started = time.perf_counter()
        ids = self.model.greedy_decode(tensor, lengths, max_length=self.max_length)
        elapsed = time.perf_counter() - started

        return PredictionResult(
            text=self.vocabulary.decode(ids[0]) if ids else "",
            inference_time=elapsed,
            model_type="attention",
            model_path=self.checkpoint_path,
        )

    def predict_text(self, image) -> str:
        return self.predict(image).text
