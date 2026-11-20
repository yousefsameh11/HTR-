"""Reusable inference module for CRNN and TrOCR.

Loads a trained checkpoint, preprocesses a handwritten line image, predicts the
transcription and measures inference time. Used by ``scripts/predict.py`` and by
the Streamlit application.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from PIL import Image

from src.data.preprocessing import CRNNPreprocessConfig, CRNNPreprocessor
from src.data.vocabulary import Vocabulary
from src.models.crnn import CRNN
from src.models.decoder import beam_search_decode, greedy_decode

ModelType = Literal["crnn", "trocr"]


@dataclass
class PredictionResult:
    text: str
    inference_time: float
    model_type: ModelType
    model_path: str


def _ensure_pil(image, mode: Optional[str] = None) -> Image.Image:
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")
    if mode is not None and image.mode != mode:
        image = image.convert(mode)
    return image


class CRNNInference:
    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: Optional[str] = None,
        img_height: Optional[int] = None,
        device: str = "auto",
        beam_width: int = 1,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.beam_width = beam_width
        self.checkpoint_path = checkpoint_path

        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            model_state = state["model_state_dict"]
            config = state.get("config") or {}
            embedded_vocab = config.get("vocabulary")
        else:  # raw state_dict
            model_state = state
            config = {}
            embedded_vocab = None

        model_cfg = dict(config.get("model") or {})
        if img_height is not None:
            model_cfg["img_height"] = img_height

        if vocab_path is not None:
            self.vocabulary = Vocabulary.load(vocab_path)
        elif embedded_vocab:
            self.vocabulary = Vocabulary(
                blank_token=embedded_vocab.get("blank_token", "<blank>"),
                unk_token=embedded_vocab.get("unk_token", " unk"),
                use_unk=embedded_vocab.get("use_unk", True),
            ).build_from_chars(embedded_vocab["chars"])
        else:
            raise ValueError(
                "No vocabulary available: pass vocab_path or use a checkpoint that "
                "stores its vocabulary in config['vocabulary']."
            )

        model_cfg.setdefault("img_height", 64)
        model_cfg.setdefault("rnn_hidden_size", 256)
        model_cfg.setdefault("rnn_num_layers", 2)
        model_cfg["num_classes"] = self.vocabulary.num_classes
        model_cfg["dropout"] = 0.0

        self.model = CRNN(**model_cfg)
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()

        self.img_height = int(model_cfg["img_height"])
        self.preprocessor = CRNNPreprocessor(
            CRNNPreprocessConfig(img_height=self.img_height)
        )

    @torch.no_grad()
    def predict(self, image) -> PredictionResult:
        image = _ensure_pil(image, "L")
        arr = self.preprocessor(image, training=False)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        logits = self.model(tensor)
        t1 = time.perf_counter()

        log_probs = torch.log_softmax(logits, dim=-1).detach()
        if self.beam_width > 1:
            indices = beam_search_decode(
                log_probs,
                beam_width=self.beam_width,
                blank_index=self.vocabulary.blank_index,
            )
        else:
            indices = greedy_decode(log_probs, blank_index=self.vocabulary.blank_index)

        return PredictionResult(
            text=self.vocabulary.indices_to_text(indices[0]),
            inference_time=t1 - t0,
            model_type="crnn",
            model_path=self.checkpoint_path,
        )


class TrOCRInference:
    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        max_length: int = 128,
        num_beams: int = 4,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.max_length = max_length
        self.num_beams = num_beams

        from src.models.trocr import TrOCRWrapper

        self.wrapper = TrOCRWrapper.load_pretrained(model_dir, device=device)

    @torch.no_grad()
    def predict(self, image) -> PredictionResult:
        image = _ensure_pil(image, "RGB")
        pixel_values = self.wrapper.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]

        t0 = time.perf_counter()
        preds = self.wrapper.generate(
            pixel_values=pixel_values,
            max_length=self.max_length,
            num_beams=self.num_beams,
        )
        t1 = time.perf_counter()

        return PredictionResult(
            text=preds[0] if preds else "",
            inference_time=t1 - t0,
            model_type="trocr",
            model_path=self.model_dir,
        )


def load_predictor(model_type: ModelType, **kwargs):
    """Factory that builds the requested predictor."""
    if model_type == "crnn":
        return CRNNInference(**kwargs)
    if model_type == "trocr":
        return TrOCRInference(**kwargs)
    raise ValueError(f"Unknown model_type: {model_type}")


def predict_image(model_type: ModelType, image, **kwargs) -> PredictionResult:
    return load_predictor(model_type, **kwargs).predict(image)
